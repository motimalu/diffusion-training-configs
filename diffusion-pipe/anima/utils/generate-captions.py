import json
import random
from pathlib import Path

INPUT_DIRS = [
    "./512x512",
    "./1024x1024",
    "./1536x1536",
]
KEEP_TAGS = 8
DROPOUT = 0.3
SHUFFLE = True
NUM_AUGMENTATIONS = 2
PROTECTED_TAGS_FILE = './protected-tags.txt'

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


def generate_captions_json(directory_path="."):
    dataset_path = Path(directory_path)
    captions_data = {}

    protected_tags = _load_protected_tags(PROTECTED_TAGS_FILE)
    if PROTECTED_TAGS_FILE and protected_tags:
        print(f"Loaded {len(protected_tags)} protected tags from {PROTECTED_TAGS_FILE}")
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    for file_path in dataset_path.iterdir():
        if file_path.suffix.lower() in image_extensions:
            image_filename = file_path.name
            base_name = file_path.stem  # e.g., 'danbooru_1'

            tag_file = dataset_path / f"{base_name}.txt"
            nl_file = dataset_path / f"{base_name}_nl.txt"
            nl_file_v2 = dataset_path / f"{base_name}_nl_v2.txt"

            if tag_file.exists() and nl_file.exists() and nl_file_v2.exists():
                try:
                    with open(tag_file, 'r', encoding='utf-8') as f:
                        tags = f.read().strip()
                    with open(nl_file, 'r', encoding='utf-8') as f:
                        nl_caption = f.read().strip()
                    with open(nl_file_v2, 'r', encoding='utf-8') as f:
                        nl_caption_v2 = f.read().strip()

                    captions = []
                    tag_list = [t.strip() for t in tags.split(',')]
                    anchor_tags = tag_list[:KEEP_TAGS]
                    tail_tags = tag_list[KEEP_TAGS:]
                    first_n_tags = ', '.join(anchor_tags)

                    for _ in range(NUM_AUGMENTATIONS):
                        SHUFFLE and random.shuffle(tail_tags)
                        tags_full = ', '.join(anchor_tags + tail_tags)

                        SHUFFLE and random.shuffle(tail_tags)
                        dropped_tail1 = [tag for tag in tail_tags if tag in protected_tags or random.random() > DROPOUT]
                        SHUFFLE and random.shuffle(tail_tags)
                        dropped_tail2 = [tag for tag in tail_tags if tag in protected_tags or random.random() > DROPOUT]


                        dropout_tags1 = ', '.join(anchor_tags + dropped_tail1)
                        dropout_tags2 = ', '.join(anchor_tags + dropped_tail2)
                        captions.extend([
                            f"{tags_full}.\n{nl_caption}",
                            f"{first_n_tags}.\n{nl_caption_v2}",
                            f"{dropout_tags1}.\n{nl_caption}",
                            f"{nl_caption_v2}\n{dropout_tags2}"
                        ])

                    captions_data[image_filename] = captions
                except Exception as e:
                    print(f"Error processing {base_name}: {e}")

    output_file = dataset_path / "captions.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(captions_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully created {output_file} with {len(captions_data)} entries.")

if __name__ == "__main__":
    for input_dir in INPUT_DIRS:
        generate_captions_json(input_dir)