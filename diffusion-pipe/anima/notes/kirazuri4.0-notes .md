
# Kirazuri (Anima) 4.0 Training Diary

Kirazuri (Anima) 4.0 is a full fine-tune of the [Anima Base v1.0 model by CircleStone Labs](https://huggingface.co/circlestone-labs/Anima/blob/main/split_files/diffusion_models/anima-base-v1.0.safetensors)

This 4.0 version is trained from the Stage 1 checkpoint used for version 3.0, repeating only the Stage 2 training with several differences:

- Updated dataset with cutoff of **29/06/2026**
- Revised training hyper-parameters
- Masked training

These changes aim to:
- Introduce new knowledge to the model
- Improve stability for anatomy and complex compositions

While maintaining the models original goals to:

- Learn new concepts/styles/characters past the base model dataset cutoff of 2025 September
- Enhance the model aesthetic guided by manually applied quality, aesthetic, and style tagging
- Improve rendering and understanding of fine-details through high-resolution training up to 1536^2

## Preamble / Disclaimers

The purpose of this article is for transparency in sharing an up-to-date overview of the training methods at the time of this writing, it is not intended to be a guide or indicative of best practices.

For reflection and error checking purposes, this article is not written with the assistance of a LLM.

The Kirazuri (Anima) model is produced in an individual hobbyist capacity with no external funding, and the model weights remain open with no additional restrictions to the base model license.

## Training Details Summary

**Trainer:** [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) [commit c1239b532031e6621fbd90aaac1c77ba99693bc3](https://github.com/tdrussell/diffusion-pipe/commit/c1239b532031e6621fbd90aaac1c77ba99693bc3)

**Training device:** NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition

**Total training time:** ~9 days

**Total samples seen(unbatched steps):** ~3,000,000

**Training resolutions:**

- 512^2
- 768^2
- 1024^2
- 1536^2

### Stage 1

See: [Kirazuri 3.0 Notes: Dataset Partitioning - Stage 1](./kirazuri3.0-notes.md#stage-1)

### Stage 2

- **Samples seen(unbatched steps):** ~1,000,000
- **Training time:** ~84 hrs
- **Learning Rate:** 2e-6
- **Learning Rate Scheduler:** Cosine
- **LLM Adaptor Learning Rate:** 2e-7
- **Precision:** Mixed BF16
- **Optimizer:** AdamW8bit with Kahan Summation
- **Weight Decay:** 0.01
- **Timestep Sampling Strategy:** Logit-Normal
- **Training Resolutions:** 512^2, 1024^2, 1536^2

### Additional Features

- Masked Training
- Tag Dropout: 30% with protected first 8 tags
- Tag Shuffle: Applied to last unprotected tags
- Natural Language: Short and Long Caption variants

## Changes from Kirazuri (Anima) v3.0

- Dataset includes recently curated 2,450 images increasing total size from 42,608 to **45,058** images
- Dataset cutoff now of **29/06/2026**
- Introduced Masked Training for images with simple backgrounds
- Updated tags+caption variants structure

## Dataset

The partitioned dataset of high-quality data originally used in Stage 2 training of version 3.0 is re-used.
To this included ~2,450 recently curated images, for a total of 18,509 images.

See: [Kirazuri 3.0 Notes: Dataset Partitioning - Stage 2](./kirazuri3.0-notes.md#dataset-partitioning---stage-2)

## Captions

[generate-captions.py](/diffusion-pipe/anima/utils/generate-captions.py) script used three files containing tags and natural language variants:
- `{filename}.txt`
- `{filename}_nl.txt`
- `{filename}_nl2.txt`

A small update to the script was made to structure the output tag combinations in order of least to most information, e.g. limited/dropout tags and short captions to full tags and captions.
The intention of this is to simulate curriculum training method is described by the Tongyi-MAI team in the paper [Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer](https://huggingface.co/papers/2511.22699).

It applied a dropout of 30% and shuffles tags into `dropout_tags1` and `dropout_tags2` keeping the first 8 tags, and a list of tags from a [protected.txt](/diffusion-pipe/anima/utils/protected-tags.txt) file.

The resulting updated `captions.json` structure for this training run appears as below:

```json
{
    "booru_1.jpg": [
      "{first_n_tags}.\n{nl_caption_v2}",
      "{nl_caption_v2}\n{dropout_tags1}",
      "{nl_caption}\n{dropout_tags2}"
      "{tags_full}.\n{nl_caption}",
    ]
}
```

## Stage 1

See: [Kirazuri 3.0 Notes: Dataset Partitioning - Stage 1](./kirazuri3.0-notes.md#stage-1-1)

## Stage 2

Stage 2 is re-trained for a full 3 epochs to complete a cosine decay from 2e-6 to 1e-6.

It targets a more balanced resolution mix of 512^2, 1024^2, and 1536^2 this time to try to prioritizes anatomical and compositional stability while also including high-resolution training.

`flux_shift` and `multiscale_loss_weight` are disabled this time.

The `llm_adaptor` is enabled for this stage, as new characters/concepts/style associations not learned in Stage 1 are introduced with the newly curated dataset.

| Resolution | Image count |
| ---------- | ----------- |
| 512^2      | 330         |
| 1024^2     | 11,213      |
| 1536^2     | 6,966       |

### Training config
- [kirazuri-v4-stage-2.toml](/diffusion-pipe/anima/configs/kirazuri-v4-stage-2.toml)
- [dataset-partition-v4-2.toml](/diffusion-pipe/anima/configs/dataset-partition-v4-s2.toml)

### Image Epoch/Steps/Samples breakdown by batch size
```
18509 images, 8 repeats from captions.json, 3 resolutions
18509*8 148072 samples/resolution/epoch
2314+6060+4644 = 13018 steps/epoch
512 Resolution: (330+11213+6966)*8 = 148072 samples / 64 batch size = 2314 steps
- iter time (s): 4.6*2314 = 2.95 hr / epoch
1024 Resolution: (11213+6966)*8 = 145432 samples / 24 batch size = 6060 steps
- iter time (s): 6.482*6060 = 10.91 hr / epoch
1536 Resolution: 6966*8 = 55728 samples / 12 batch size = 4644 steps
- iter time (s): 4.837*4644 = 6.23 hr / epoch
Total 2.95+10.91+6.23 = 20.09 hr/epoch, max VRAM utilization ~71 GB
Total Samples seen(unbatched steps): (55728+145432+148072)*3 = ~1,047,696
```

# Results and future considerations

## Training curriculum planning

The curriculum training method is described by the Tongyi-MAI team in the paper [Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer](https://huggingface.co/papers/2511.22699)

This time the model saw the three caption variants - tags, short NL, long NL - incrementally for each epoch, which roughly simulates the curriculum approach.

If using more stages where training is continued, splitting the caption variants across those stages might also be possible.

Not sure if this would necessarily yield better results though.

## High-resolution training

Dropping 1280^2 and substituting with 512^2 resolution yielded better results.

Maybe this is because of how the base Cosmos 2 and Anima models are also heavily pre-trained at the 512^2 resolution, including it with a high batch seems to always be beneficial.

It's not clear whether dropping `flux_shift` and/or `multiscale_loss_weight`, or including the 512^2 resolution helped the most, but the result does appear to be better.

## Masked training

The results from some LoRA training tests showed positive results from masked training of simple background images, which is why it was decided to include it with this training of a full-finetune.

Background and composition control seem to be improved, and character details were also learned well enough, so it seems worth continuing to include `mask_path` datasets when training many images with simple backgrounds.