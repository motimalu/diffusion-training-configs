
# Kirazuri (Anima) 3.0 Training Diary

Kirazuri (Anima) 3.0 is a full fine-tune of the [Anima Base v1.0 model by CircleStone Labs](https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/diffusion_models/anima-base-v1.0.safetensors) focused on several goals:

- Learn new concepts/styles/characters past the base model dataset cutoff of 2025 September
- Enhance the model aesthetic guided by manually applied quality, aesthetic, and style tagging
- Improve rendering and understanding of fine-details through high-resolution training for 1024^24, 1280^2, and 1536^2

## Preamble / Disclaimers

The purpose of this article is for transparency in sharing an up-to-date overview of the training methods at the time of this writing training methods employed, it is not intended to be a guide or indicative of best practices.

For reflection and error checking purposes, this article is not written with the assistance of a LLM.

The Kirazuri (Anima) model is produced in an individual hobbyist capacity with no external funding, and the model weights remain open with no additional restrictions to the base model license.

## Training Details Summary

**Trainer:** [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) [commit b0aa4f1e03169f3280c8518d37570a448420f8be](https://github.com/tdrussell/diffusion-pipe/commit/b0aa4f1e03169f3280c8518d37570a448420f8be)

**Training device:** NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition

**Total training time:** ~10 days

**Total samples seen(unbatched steps):** ~2,550,000

**Training resolutions:**

- 512^2
- 768^2
- 1024^2
- 1280^2
- 1536^2

### Stage 1

- **Samples seen(unbatched steps):** ~2,000,000
- **Training time:** ~125 hrs
- **Learning Rate:** 6e-6
- **Learning Rate Scheduler:** Cosine
- **LLM Adaptor Learning Rate:** 8e-7
- **Precision:** Mixed BF16
- **Optimizer:** AdamW8bit with Kahan Summation
- **Weight Decay:** 0.01
- **Timestep Sampling Strategy:** Logit-Normal

### Stage 2

- **Samples seen(unbatched steps):** ~550,000
- **Training time:** ~118 hrs
- **Learning Rate:** 3e-6
- **Learning Rate Scheduler:** Cosine
- **LLM Adaptor Learning Rate:** 0
- **Flux Shift:** Enabled
- **Multi-Scale Loss Weight:** 0.5
- **Precision:** Mixed BF16
- **Optimizer:** AdamW8bit with Kahan Summation
- **Weight Decay:** 0.01
- **Timestep Sampling Strategy:** Logit-Normal

### Additional Features

- Tag Dropout: 30% with protected first 8 tags
- Tag Shuffle: Applied to last unprotected tags
- Natural Language: Short and Long Caption variants

## Changes from Kirazuri (Anima) v2.0

- Dataset includes recently curated 7,071 images increasing total size from 35,537 to **42,608** images
- Dataset cutoff now of **2026/05/12**.
- Trained at 5 total resolutions in two-stage training
  - Stage 1 - 512^2, 768^2, 1024^2
  - Stage 2 - 1024^2, 1280^2 1536^2
- Introduced cosine learning rate scheduler for smooth learning rate transition between training stages
- Re-captioned full dataset for a second natural language captions variant with updated captioning script

## Dataset

The dataset used to train this model includes almost all datasets curated for previous full fine-tunes and various LoRA and over several years, starting when tagging and data storage practices were improved in December 2024.

These are all available on the CivitAI account posting this article, so they should also serve as a good reference of the models capabilities in terms of what is trained.

### Methodology of dataset preparation

Filtering and selecting only valuable data is the first priority.

The concept that flawed, biased or poor quality input produces a result or output of similar quality - "garbage-in, garbage-out" is something generally adhered to for the datasets curation.

A huge amount of data is not really necessary for learning additional individual concepts like a style, a characters likeness, or an outfit.

Roughly one in one hundred images assessed for training a given purpose like the above may be included in the total datasets curation.

While the dataset cutoff is 2026/05/12, this is not all inclusive as a manually curated dataset.

### What is filtered
- Works that directly expresses the intent to not be used for AI training
  - "No AI" disclaimers or Poisoned images
  - Both are avoided to respect artists intent with their creations
- AI Generated and Synthetic data
  - Images with Booru tags "ai-assisted", "ai-generated" are automatically excluded
  - Images that are visually obviously AI generated I have also filtered
  - Indirect distillation of another output is not desired
- Photo Realistic depictions of real people
  - With the exception of photo backgrounds with 2d compositing, sought to improve backgrounds quality
- Images that would be considered low quality without having other valuable aspects, e.g.
  - A sketch of a well-represented character already known by most base models would not be included
- Simple backgrounds
  - Images with simple backgrounds are generally avoided unless of high quality or expressing rare concepts
- Heavily watermarked images

### Image Quality and Aesthetic rating

After filtering, images are manually rated for quality and aesthetic modifiers.

Quality modifiers:
- `masterpiece`
- `best quality`
- `low quality`

Aesthetic modifiers:
- `very aesthetic` 
- `aesthetic`

### Image modification
- Modifications to images are kept as minimal as possible.
- Most only cropping images to remove negative space is applied.
Textual elements including artist signatures are generally preserved.
  - If these are labelled accurately, they should be a benefit to the models understanding and text rendering capabilities.
  - If an artist name is prompted in isolation, it is appropriate that their signature would also be generated if it is a prominent feature in their works.

### Booru style tagging

Images from Booru data sources have their full tags preserved.

SmilingWolf/wd-eva02-large-tagger-v3 was used for additional image Booru style tagging.

### Natural language captioning

Booru style tags and metadata context are used to provide grounding for natural language description generation.

Two variants of natural language captions were generated for the full dataset using models:
- `Qwen/Qwen3.5-122B-A10B`
- `Qwen/Qwen3.6-35B-A3B`

## Dataset Partitioning - Stage 1

This first stage contained the entire dataset of 42608, all images are partitioned into three folders by their total pixel areas using [partition-resolutions.py](/diffusion-pipe/anima/utils/partition-resolutions.py).

- 512^2 - 512 images
- 768^2 - 7008 images
- 1024^2 - 35088 images

Then generated captions.json from `.txt` and `_nl*.txt` sidecar files in each directory with [generate-captions.py](/diffusion-pipe/anima/utils/generate-captions.py).

## Dataset Partitioning - Stage 2

This second stage filtered the dataset first for high resolution images by total pixel areas using [partition-resolutions.py](/diffusion-pipe/anima/utils/partition-resolutions.py):

- 1536^2 - 8,085 images
- 1280^2 - 2,750 images
- 1024^2 - 7,041 images

Selected for further training based on and the manual quality and aesthetic ratings and date:

| Classification    | Query                                                                     | Images |
| ----------------- | ------------------------------------------------------------------------- | ------ |
| rated-masterpiece | "masterpiece"                                                             | 205    |
| rated-high        | "best quality" AND "very aesthetic"                                       | 3969   |
| rated-new         | "date > 2025/10/01" AND "best quality" OR "very aesthetic" OR "aesthetic" | 2867   |

Dropped at this stage:

| Classification | Query                                                                     | Images |
| -------------- | ------------------------------------------------------------------------- | ------ |
| rated-old      | "date < 2025/10/01" AND "best quality" OR "very aesthetic" OR "aesthetic" | 9761   |
| unrated-old    | "date < 2025/10/01"                                                       | 6123   |
| unrated-new    | "date > 2025/10/01"                                                       | 1328   |

rated-new, rated-high, rated-masterpiece are kept, and rated-old, unrated-old, unrated-new are dropped.

Again generated captions.json from `.txt` and `_nl*.txt` sidecar files in each new directory with with [generate-captions.py](/diffusion-pipe/anima/utils/generate-captions.py).

## Captions

[generate-captions.py](/diffusion-pipe/anima/utils/generate-captions.py) script used three files containing tags and natural language variants:
- `{filename}.txt`
- `{filename}_nl.txt`
- `{filename}_nl2.txt`

Applied a dropout of 30% and shuffles tags into `dropout_tags1` and `dropout_tags2` keeping the first 8 tags, and a list of tags from a [protected.txt](/diffusion-pipe/anima/utils/protected-tags.txt) file.

This dropout, shuffle, and protected tag approach is adapted from @bluvoll fork of diffusion-pipe, which would apply the transformations at runtime.

It is adapted this way to use `captions.json` which was intended to allow for:
- captions pre-caching, shifting the performance overhead to memory limitation and caching time
- no required changes/fork of the diffusion-pipe repository

```json
{
    "booru_1.jpg": [
      "{tags_full}.\n{nl_caption}",
      "{first_n_tags}.\n{nl_caption_v2}",
      "{dropout_tags1}.\n{nl_caption}",
      "{nl_caption_v2}\n{dropout_tags2}"
    ]
}
```

## Stage 1

Stage 1 was trained for 2 of the total 4 epochs in the training settings.

Purpose of this was to use the cosine scheduler decay from 6e-6 to 3e-6, to smoothly transition to the 3e-6 starting LR to be used in Stage 2.

| Resolution | Image count |
| ---------- | ----------- |
| 512^2      | 512         |
| 768^2      | 7,008       |
| 1024^2     | 35,088      |

### Training config
- [kirazuri-v3-stage-1.toml](/diffusion-pipe/anima/configs/kirazuri-v3-stage-1.toml)
- [dataset-partition-s1.toml](/diffusion-pipe/anima/configs/dataset-partition-s1.toml)

### Image Epoch/Steps/Samples breakdown by batch size
```
42608 images, 8 repeats from captions.json, 3 resolutions
42608*8 340864 samples/resolution/epoch
5326+10524+17544 = 33394 steps/epoch
512 Resolution: (512+7008+35088)*8 = 340864 samples / 64 batch size = 5326 steps
768 Resolution: (7008+35088)*8 = 336768 samples / 32 batch size = 10524 steps
1024 Resolution: 35088*8 = 280704 samples / 16 batch size = 17544 steps
```

## Stage 2

Stage 2 is trained for a full 3 epochs to complete a cosine decay from 3e-6 to 1e-6.

It targets the higher resolutions 1024^2, 1280^2, and 1536^2 to learn fine details.

`flux_shift` and `multiscale_loss_weight` are enabled for this stage, as they are assumed to be beneficial for higher-resolution training.

The `llm_adaptor` is disabled for this stage, as it has already learned new characters/concepts/style associations in Stage 1.

| Resolution | Image count |
| ---------- | ----------- |
| 1024^2     | 7,041       |
| 1280^2     | 2,750       |
| 1536^2     | 8,085       |

### Training config
- [kirazuri-v3-stage-2.toml](/diffusion-pipe/anima/configs/kirazuri-v3-stage-2.toml)
- [dataset-partition-s2.toml](/diffusion-pipe/anima/configs/dataset-partition-s2.toml)

### Image Epoch/Steps/Samples breakdown by batch size
```
16106 images, 8 repeats from captions.json, 3 resolutions
16106*8 128848 samples/resolution/epoch
3521+4533+6945 = 14999 steps/epoch
1024 Resolution: 7041*8 = 56328 samples / 16 batch size = 3521 steps
1280 Resolution: (2120+6945)*8 = 72520 samples / 16 batch size = 4533 steps
1536 Resolution: 6945*8 = 55560 samples / 8 batch size = 6945 steps
```

# Results and future considerations

## Training curriculum planning

The curriculum training method is described by the Tongyi-MAI team in the paper [Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer](https://huggingface.co/papers/2511.22699)

While this model used three caption variants - tags, short NL, long NL - they were trained with random selection for each stage.

A future curriculum training approach could be possibly be implemented by training first tags only, then tags and short NL, and finally tags and long NL.

## High-resolution training

The DiT architecture generalizes well from the lower-res training to higher resolutions.

So the additional training time and cost associated with the higher-resolution training may not be entirely justified.

Fine details were better picked up and learned from the higher res training however.

## Masked loss

While simple backgrounds are generally avoided and filtered where possible from the dataset, it may be worth applying to the remaining images.
