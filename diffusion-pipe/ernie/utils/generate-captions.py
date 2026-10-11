import os
import re
import json
import random
from pathlib import Path

INPUT_DIRS = [
    "./1_collection_name4/0_update/512x512/",
    "./1_collection_name4/0_update/768x768/",
    "./1_collection_name4/768x768-rated/",
    "./1_collection_name4/0_update/1024x1024/",
    "./1_collection_name4/1024x1024-rated/",
    "./1_collection_name4/1024x1024-rated-high/",
]

KEEP_TAGS = 8
DROPOUT = 0.1           # Per-tag Bernoulli drop probability for 'dropout' tag mode
                        # (protected tags are never dropped).
MAX_CAPTION_CHARS = 3000  # Hard cap on each combined caption string
SHUFFLE = True
NUM_AUGMENTATIONS = 1   # 1 -> exactly 6 variants per image
PROTECTED_TAGS_FILE = './protected-tags.txt'
SEED = 42               # None = non-deterministic. Fixed seed keeps re-runs reproducible.

# The 6 caption variants per image: (tag_mode, nl_slot, order)
#   tag_mode: 'anchors' = first KEEP_TAGS only | 'dropout' = Bernoulli-dropped tail
#             'full'    = every tag, original order
#   nl_slot:  1..6 -> _nl, _nl_v2 ... _nl_v6
#   order:    'tags_nl' = "tags.\n<nl>" | 'nl_tags' = "<nl>\ntags"
VARIANTS = [
    ('anchors', 1, 'tags_nl'),  # first_n_tags + _nl
    ('dropout', 3, 'nl_tags'),  # _nl_v3 + dropout_tags1
    ('dropout', 5, 'tags_nl'),  # dropout_tags2 + _nl_v5
    ('dropout', 2, 'nl_tags'),  # _nl_v2 + dropout_tags3
    ('dropout', 4, 'tags_nl'),  # dropout_tags4 + _nl_v4
    ('full',    6, 'tags_nl'),  # tags_full + _nl_v6
]

NL_FILE_SUFFIX = {1: '_nl', 2: '_nl_v2', 3: '_nl_v3', 4: '_nl_v4', 5: '_nl_v5', 6: '_nl_v6'}
TERM_PUNCT = ('.', '!', '?', '。', '！', '？')


def _load_protected_tags(filepath):
    if not filepath:
        return set()

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tags = set()
            for line in f:
                tag = line.strip()
                if tag and not tag.startswith('#'):  # Allow comments
                    tags.add(tag)
            return tags
    except FileNotFoundError:
        print(f"Warning: protected_tags_file not found: {filepath}")
        return set()
    except Exception as e:
        print(f"Warning: Error loading protected_tags_file: {e}")
        return set()


def split_sentences(text):
    """Split NL text into sentences (keeps terminal punctuation attached)."""
    parts = re.split(r'(?<=[.!?。！？])\s+', text.strip())
    return [p for p in (s.strip() for s in parts) if p]


def _fill_nl(sentences, nl_budget):
    """Keep front sentences in order while they fit; if even the first doesn't
    fit, truncate it at a word boundary (last resort)."""
    kept, cur = [], 0
    for s in sentences:
        add = len(s) + (1 if kept else 0)
        if cur + add <= nl_budget:
            kept.append(s)
            cur += add
        else:
            break
    if not kept and sentences:
        s0 = sentences[0]
        if len(s0) <= nl_budget:
            kept = [s0]
            cur = len(s0)
        elif nl_budget > 0:
            cut = s0[:nl_budget]
            if ' ' in cut:
                cut = cut.rsplit(' ', 1)[0]
            kept = [cut.rstrip(',;: ').strip() or s0[:1]]
            cur = len(kept[0])
    return kept, cur


def build_variant(tag_mode, tag_list, nl_text, order, rng, protected_tags):
    """Build one caption variant under MAX_CAPTION_CHARS.

    Cap policy (same as the Stage-1 generator, applied to the combined string):
    - NL is cut sentence-by-sentence from the end
    - if even the first sentence doesn't fit, droppable tail tags are removed
      (last-in-first-out; anchors always kept) until it does; truncating the
      first sentence at a word boundary is only the last resort
    Returns (caption, nl_sentences_removed, first_sentence_truncated, tags_trimmed).
    """
    anchor = tag_list[:KEEP_TAGS]
    tail = tag_list[KEEP_TAGS:]
    droppable = []  # tail tags removable under cap pressure, removed from the end
    if tag_mode == 'anchors':
        tags_str = ', '.join(anchor)
    elif tag_mode == 'full':
        tags_str = ', '.join(tag_list)
        droppable = list(tail)
    else:  # 'dropout': per-tag Bernoulli (protected tags always kept)
        t = list(tail)
        if SHUFFLE:
            rng.shuffle(t)
        kept_tail = [x for x in t if x in protected_tags or rng.random() > DROPOUT]
        tags_str = ', '.join(anchor + kept_tail)
        droppable = kept_tail

    has_nl = bool(nl_text)
    sentences = split_sentences(nl_text) if has_nl else []
    sep = ('.\n' if order == 'tags_nl' else '\n') if has_nl else ''
    first_len = len(sentences[0]) if sentences else 0

    # If the first sentence doesn't fit, free room by trimming tail tags first
    tags_trimmed = 0
    if has_nl and first_len > MAX_CAPTION_CHARS - len(tags_str) - len(sep) and droppable:
        while droppable:
            droppable.pop()
            tags_trimmed += 1
            cand = ', '.join(anchor + droppable)
            if len(cand) + len(sep) + first_len <= MAX_CAPTION_CHARS:
                tags_str = cand
                break
        else:
            tags_str = ', '.join(anchor)

    kept, _ = _fill_nl(sentences, MAX_CAPTION_CHARS - len(tags_str) - len(sep))
    nl_removed = len(sentences) - len(kept)
    first_trunc = bool(kept) and kept[0] != sentences[0]

    if has_nl:
        nl_kept = ' '.join(kept)
        if order == 'nl_tags' and nl_kept and not nl_kept.endswith(TERM_PUNCT):
            nl_kept += '.'
        caption = (tags_str + sep + nl_kept) if order == 'tags_nl' else (nl_kept + sep + tags_str)
    else:
        caption = tags_str

    if len(caption) > MAX_CAPTION_CHARS:
        raise RuntimeError(f"cap violation: {len(caption)} > {MAX_CAPTION_CHARS} "
                           f"(mode={tag_mode}, order={order})")
    return caption, nl_removed, first_trunc, tags_trimmed


def resolve_nl_slot(available, slot):
    """Prefer the requested variant; fall back cyclically to the nearest other
    available variant (slot, slot+1, ..., wrapping 1..6). None if no NL exists."""
    for i in range(len(NL_FILE_SUFFIX)):
        v = ((slot + i - 1) % 6) + 1
        if v in available:
            return v
    return None


def generate_captions_json(directory_path="."):
    dataset_path = Path(directory_path)
    captions_data = {}

    protected_tags = _load_protected_tags(PROTECTED_TAGS_FILE)
    if PROTECTED_TAGS_FILE and protected_tags:
        print(f"Loaded {len(protected_tags)} protected tags from {PROTECTED_TAGS_FILE}")

    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    rng = random.Random(SEED)

    all_lens = []
    n_dedup = 0
    n_nl_cut = 0
    n_first_trunc = 0
    n_tags_trimmed = 0
    n_no_nl = 0            # images with no usable NL variant (tag-only captions)
    n_skipped = 0          # images with no tag file
    fallbacks = {v: 0 for v in range(1, 7)}  # slot -> times a different variant was used

    for file_path in sorted(dataset_path.iterdir()):
        if file_path.suffix.lower() in image_extensions:
            image_filename = file_path.name
            base_name = file_path.stem  # e.g., 'danbooru_1'

            tag_file = dataset_path / f"{base_name}.txt"
            if not tag_file.exists():
                n_skipped += 1
                continue

            # Read whichever NL variants exist (missing/empty ones fall back below)
            available = {}
            for v, suffix in NL_FILE_SUFFIX.items():
                f = dataset_path / f"{base_name}{suffix}.txt"
                if f.exists():
                    text = f.read_text(encoding='utf-8').strip()
                    if text:
                        available[v] = text

            try:
                tags = tag_file.read_text(encoding='utf-8').strip()
                tag_list = [t.strip() for t in tags.split(',') if t.strip()]

                if not available:
                    n_no_nl += 1

                captions = []
                for _ in range(NUM_AUGMENTATIONS):
                    for tag_mode, slot, order in VARIANTS:
                        used = resolve_nl_slot(available, slot)
                        if used is None:
                            nl_text = ''
                        elif used != slot:
                            fallbacks[slot] += 1
                            nl_text = available[used]
                        else:
                            nl_text = available[slot]

                        caption, nl_removed, first_trunc, tags_trimmed = build_variant(
                            tag_mode, tag_list, nl_text, order, rng, protected_tags)
                        if caption not in captions:  # dedupe (happens when fallbacks collide)
                            captions.append(caption)
                        else:
                            n_dedup += 1
                        all_lens.append(len(caption))
                        if nl_removed >= 1:
                            n_nl_cut += 1
                        if first_trunc:
                            n_first_trunc += 1
                        if tags_trimmed >= 1:
                            n_tags_trimmed += 1

                captions_data[image_filename] = captions
            except Exception as e:
                print(f"Error processing {base_name}: {e}")

    output_file = dataset_path / "captions.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(captions_data, f, indent=2, ensure_ascii=False)

    n_variants = sum(len(c) for c in captions_data.values())
    print(f"Successfully created {output_file} with {len(captions_data)} entries "
          f"({n_variants} captions, {n_dedup} duplicates removed).")
    if all_lens:
        all_lens.sort()
        q = lambda p: all_lens[min(len(all_lens) - 1, int(len(all_lens) * p))]
        print(f"Caption length: p50={q(0.50)} p90={q(0.90)} p99={q(0.99)} max={all_lens[-1]} (cap={MAX_CAPTION_CHARS})")
        print(f"NL sentence(s) cut: {n_nl_cut}  |  NL first-sentence truncated: {n_first_trunc}  |  tag tail trimmed: {n_tags_trimmed}")
        fb = '  '.join(f'v{v}:{fallbacks[v]}' for v in range(1, 7))
        print(f"NL fallbacks (slot -> other variant used): {fb}  |  images with no NL at all: {n_no_nl}  |  images skipped (no tag file): {n_skipped}")


if __name__ == "__main__":
    for input_dir in INPUT_DIRS:
        generate_captions_json(input_dir)
