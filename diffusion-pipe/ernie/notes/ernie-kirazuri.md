
# Kirazuri (Ernie) Training Diary

Kirazuri (Ernie) is a full fine-tune of the ERNIE-Image model. 
[ERNIE-Image is an open text-to-image generation model developed by the ERNIE-Image team at Baidu](https://huggingface.co/baidu/ERNIE-Image)

Version 0.1 (Latest) is trained on ~50,000 images at multiple resolutions in three total stages up to a maximum resolution of 1024^2 pixels.

This fine-tune focuses on several goals:

- Learn new concepts/styles/characters associated with tag-based prompting.
- Enhance the model aesthetic guided by manually applied quality, aesthetic, and style tagging
- Preserve the text-rendering capabilities in multiple languages (English, Chinese, Japanese)

## Preamble / Disclaimers

The purpose of this article is for transparency in sharing an up-to-date overview of the training methods at the time of this writing, it is not intended to be a guide or indicative of best practices.

For reflection and error checking purposes, this article is not written with the assistance of a LLM.

The Kirazuri (Ernie) model is produced in an individual hobbyist capacity with no external funding, the model weights remain open with no additional restrictions to the base model license - Apache License 2.0.

## Training Details Summary

**Trainer:** [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)

**Training device:** NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition

**Total training time:** ~500 hrs (~20 days)

**Total samples seen(unbatched steps):** ~1,328,000

**Training resolutions:**

- 512^2
- 768^2
- 1024^2

### Stage 1

- **Samples seen(unbatched steps):** ~600,000
- **Training time:** ~104 hrs
- **Learning Rate:** 6e-6
- **Learning Rate Scheduler:** Cosine
- **Precision:** Mixed BF16
- **Optimizer:** AdamW8bit with Kahan Summation
- **Weight Decay:** 0.01
- **Timestep Sampling Strategy:** Logit-Normal
- **Shift:** 3

### Stage 2

- **Samples seen(unbatched steps):** ~468,000
- **Training time:** ~181 hrs
- **Learning Rate:** 4e-6
- **Learning Rate Scheduler:** Cosine
- **Precision:** Mixed BF16
- **Optimizer:** AdamW8bit with Kahan Summation
- **Weight Decay:** 0.01
- **Timestep Sampling Strategy:** Logit-Normal
- **Shift:** 4

### Stage 3

- **Samples seen(unbatched steps):** ~260,000
- **Training time:** ~215 hrs
- **Learning Rate:** 4e-6
- **Learning Rate Scheduler:** Cosine
- **Precision:** Mixed BF16
- **Optimizer:** AdamW8bit with Kahan Summation
- **Weight Decay:** 0.01
- **Timestep Sampling Strategy:** Logit-Normal
- **Shift:** 4

### Additional Features

- Tag Dropout: 10% with protected first 8 tags
- Tag Shuffle: Applied to last unprotected tags
- Natural Language: 6 total Short and Long Caption variants

## Dataset

The dataset is a continuation of works curated for previous full-finetuning under the Kirazuri model family.
With the addition of new data with a cutoff date of **2026/09/24** curated to a total of **50,739** images for this finetune.

See: [Kirazuri 3.0 Notes: Dataset](./kirazuri3.0-notes.md#Dataset)


### Methodology of dataset preparation

With the compositional and text-generation capabilities of ERNIE-Image being what is considered the stronger point, new data that complements these capabilities was sought out for inclusion in the training dataset.

Text-generation:
- Manga panels
- Magazine covers
- Dialogue boxes

Composition:
- Multi-person
- Detailed backgrounds
- Poses

## Captions

For captioning of the dataset, several modifications were made to the captioning scripts that now uses:

- One file with the image tags in booru format
- Six total natural language prompt variants; three short, three long
- Dedicated OCR model processing of all images to detect and label text
- Metadata referencing for booru sourced images

Natural Language captioning script has been updated to now reference several files when captioning a given image:

1. `{filename}.txt`         - image tags
2. `{filename}_meta.json`   - booru image file metadata
3. `{filename}_ocr.txt`     - sanitized OCR model output of detected text

[generate-captions.py](/diffusion-pipe/ernie/utils/generate-captions.py) script used 7 files to build captions for each image:
1. `{filename}.txt`         - image tags
2. `{filename}_nl.txt`      - long natural language variant
3. `{filename}_nl_v2.txt`   - short natural language variant
4. `{filename}_nl_v3.txt`   - long natural language variant
5. `{filename}_nl_v4.txt`   - short natural language variant
6. `{filename}_nl_v5.txt`   - long natural language variant
7. `{filename}_nl_v6.txt`   - short natural language variant

A dropout of 10% and tag shuffing is applied keeping the first 8 tags, and avoiding dropping any tags from a [protected.txt](/diffusion-pipe/ernie/utils/protected-tags.txt) file.

The shuffle and dropout applied tag variants are combined with the natural language variants alternatively at the start or end of the prompt, separated by a period and newline.

Combined tags+natural language captions for each image were truncated to maintain less than 3000 characters.

The resulting updated `captions.json` structure is as below:

```json
{
    "filename_1.jpg": [
      "{first_n_tags}.\n{nl_caption_1}",
      "{nl_caption_2}\n{dropout_tags_1}",
      "{dropout_tags_2}.\n{nl_caption_3}",
      "{nl_caption_4}\n{dropout_tags_3}",
      "{dropout_tags_4}.\n{nl_caption_5}",
      "{tags_full}.\n{nl_caption_6}",
    ]
}
```

## Stage 1

This stage was exploratory training at 512^2 to test the models convergence on the new data.

| Resolution | Image count |
| ---------- | ----------- |
| 512^2      | 49879       |

### Image Epoch/Steps/Samples breakdown

Epochs: 2
Total images = 858+7772+14786+5435+11047+9981 = 49879
Total entries (images*resolutions) = 49879
Repeats from captions.json = 6
Total samples/epoch 49879*6 = 299274
Wall-clock training time: 4.332 days

### Training config
- [ernie-kirazuri-s1.toml](/diffusion-pipe/ernie/configs/ernie-kirazuri-s1.toml)
- [ernie-kirazuri-dataset-s1.toml](/diffusion-pipe/ernie/configs/ernie-kirazuri-dataset-s1.toml)

## Stage 2

Stage 2 is initiated as a new training run from the Stage 1 epoch 2 model output.

860 additional images were curated while Stage 1 training was underway, and included.

| Resolution | Image count |
| ---------- | ----------- |
| 512^2      | 50,739      |
| 768^2     | 5,819       |
| 1024^2     | 21,497      |

Images were partitioned by their resolution sizes.

And further partitioned by their human-applied quality tags.

To filter only high-quality, high-resolution images for the higher resolution training.

### Image Epoch/Steps/Samples breakdown

Epochs: 1
Total images = 50,739 (49,879 base + 860 in 0_update)
Total entries (images*resolutions) = 78,055
Repeats from captions.json = 6
Total samples/epoch = 78,055 entries*6 = 468,330
Wall-clock training time: 7.574 days

### Training config
- [ernie-kirazuri-s2.toml](/diffusion-pipe/ernie/configs/ernie-kirazuri-s2.toml)
- [ernie-kirazuri-dataset-s2.toml](/diffusion-pipe/ernie/configs/ernie-kirazuri-dataset-s2.toml)

## Stage 3

Stage 3 is initiated as a new training run from the Stage 2 epoch 2 model output.

This stage is used for 1024^2 only training, and reduces the dataset to only to high-resolution, high-quality filtered datasets:

1024x1024-rated
1024x1024-rated-high

And greater `N_update` datasets:

0_update/1024x1024
1_update/

| Resolution | Image count |
| ---------- | ----------- |
| 1024^2     | 21,626      |

### Image Epoch/Steps/Samples breakdown

Epochs: 2
Total images = 469 + 129 + 11047 + 9981 = 21626
Total entries (images*resolutions) = 21626
Repeats from captions.json = 6
Total samples/epoch = 21629 * 6 = 129756
Wall-clock training time: 8.949 days

### Training config
- [ernie-kirazuri-s3.toml](/diffusion-pipe/ernie/configs/ernie-kirazuri-s3.toml)
- [ernie-kirazuri-dataset-s3.toml](/diffusion-pipe/ernie/configs/ernie-kirazuri-dataset-s3.toml)

# Results and future considerations

**Reflecting on the finetuning goals:**

> - Learn new concepts/styles/characters associated with tag-based prompting.

Convergence on tags with >100 image representations appeared, model has harder time with lower counts and still learning general tag meanings.
Cannot for example separate multi-subject likeness by tags alone.
Aiming to reach a level of being able to reproduce concepts with only ~15 images of representation, as was possible with previous finetunes.

> - Enhance the model aesthetic guided by manually applied quality, aesthetic, and style tagging

Model has already shifted noticeably from the base model aesthetic and learned art style tags with >100 image representations.

> - Preserve the text-rendering capabilities in multiple languages (English, Chinese, Japanese)

No noticeable degradation of text-rendering capabilities so far.

To fully meet the original goals, more training is required, hence the 0.1 minor release version as an intermediate checkpoint.

Training a model of this size is a massive undertaking for the single-gpu training used so far (20 wall-clock days of training), so now considering renting multi-gpu clusters for further training.

Hoping that with the amount of training so far, tag-based concepts can be more easily picked up with less training from the current checkpoint - may test this with LoRA training.

## Prompt Enhancing

ERNIE-Image ships with a Prompt Enhancer that is optimized for the models captioning style - Natural Language in Chinese or English.
This PE defaults to Chinese-language outputs, and as such is not ideal for this finetune that is only trained with English tags+captions.

As one of ERNIE-Images biggest strengths is the capability to follow long prompts generated by the PE, training a similar PE optimized for this finetune's captioning format of tags+natural language is planned.

## Training curriculum planning

ERNIE-Image differs from the Z-Image gradually increasing complexity ladder for captions, by starting with high-context captions from the early pre-training stage.

## High-resolution training

A Stage 4 with high-resolution training at 1280^2 and 1536^2 resolutions is planned, though the ERNIE-Image based model is not trained at these resolutions.

The appropriate Flux Shift values to use for these resolutions needs to be determined.

Based on Ernie Image's scheduler_config.json (shift: 4.0, use_dynamic_shifting: false).
https://huggingface.co/baidu/ERNIE-Image/blob/main/scheduler/scheduler_config.json

For multi-res training at higher resolutions, could scale the time-shift linearly with token count as with SD3 to match this.
i.e. `shift = 4.0 * (tokens / 4096.0)` hitting shift=9.0 at 1536^2.

Or use the flux 1 shallow exponential curve, hitting shift=~7.51 at 1536^2.

### Masked training

Masked training showed improvements in previous full-finetuning, but has not yet been tested with ERNIE-Image.

Previous mask generation tooling had a high failure rate such that I could only use some of the output.

New models which can produce high-quality masks and depth-maps such as [huawei-bayerlab/marigold-v2-0](https://huggingface.co/huawei-bayerlab/marigold-v2-0) seem like they would be work trying. 

Guiding attention to ignore empty backgrounds on images, particularly for the compute-expensive high resolution training seems worth exploring.
