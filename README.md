# CSM-VL: Conditional Subspace Matryoshka for Vision-Language Embeddings

This repository trains vision-language retrieval encoders with Matryoshka
representations. In addition to the existing fixed-prefix MRL and ESE losses,
it includes **CSM-VL (Conditional Subspace Matryoshka for Vision-Language
Embeddings)**: a router selects an input-specific order of embedding groups,
so every prefix is nested while its active dimensions can differ across
examples.

## Setup and data

```bash
python -m venv vlm
source vlm/bin/activate
pip install -r requirements.txt

bash download_traindata.sh
bash download_traindata_2.sh
````

Evaluation images can optionally be downloaded from the
[`TIGER-Lab/MMEB-eval`](https://huggingface.co/datasets/TIGER-Lab/MMEB-eval)
dataset. If your Transformers installation has the known Qwen2-VL image
processor issue, run `python fix_lib.py`.

## CSM-VL

The implementation partitions the embedding into equal-sized, disjoint
groups. For each query, a lightweight MLP router ranks the groups. Selecting
the first `k` entries of that ranking creates a strictly nested subspace at
each budget.

The router consumes per-group activation statistics (mean, variance, and RMS),
so its parameters have a fixed shape across backbones and are initialized
before DistributedDataParallel starts.

The training objective combines:

1. **Base loss (`L_base`)** — InfoNCE at every routed prefix.
2. **Utility-guided routing (`L_UG`)** — cross-entropy supervision from the
   detached marginal InfoNCE reduction of each available group.
3. **Conditional multimodal interaction (`L_CMI`)** — discourages router mass
   on groups with redundant residual interaction. It uses the paired query and
   target retrieval views, avoiding model-specific additional forward passes.

The full objective is
`L_base + 0.45 * L_UG + 0.55 * L_CMI`. These are the default values of
`cms_utility_weight` and `cms_cmi_weight`, respectively. The training log
includes `cms_base_loss`, `cms_ug_loss`, and `cms_cmi_loss` for analysis.

### Configuration

Use one of the following loss names with `--kd_loss_type`:

| Loss name | Enabled terms   | Purpose                              |
| --------- | --------------- | ------------------------------------ |
| `cms_mrl` | Base + UG + CMI | Full CSM-VL model.                   |
| `cms_ug`  | Base + UG       | Utility-guided-routing ablation.     |
| `cms_cmi` | Base + CMI      | Interaction-regularization ablation. |

Important options are `--cms_num_groups` (default `8`),
`--cms_router_hidden_dim` (default `256`), `--cms_utility_temperature`,
`--cms_utility_weight` (default `0.45`), and `--cms_cmi_weight` (default
`0.55`). The embedding dimension does not need to be divisible by the group
count; CSM-VL pads the final group only while computing the loss.

### Backbone optimization settings

All CSM-VL scripts use the following settings. Qwen3-VL-8B intentionally
follows the Qwen3-VL-2B optimization configuration.

| Setting                 |  FastVLM-0.5B |   B3-Qwen2-2B |   Qwen3-VL-2B |   Qwen3-VL-8B |
| ----------------------- | ------------: | ------------: | ------------: | ------------: |
| Epochs                  |             1 |             1 |             1 |             1 |
| Learning rate           |        `1e-4` |        `1e-4` |        `1e-4` |        `1e-4` |
| Projector learning rate |        `5e-4` |        `5e-4` |        `5e-4` |        `5e-4` |
| Batch size              |            32 |            16 |            16 |            16 |
| Scheduler / warmup      | Cosine / 0.03 | Cosine / 0.03 | Cosine / 0.03 | Cosine / 0.03 |
| Weight decay            |          0.01 |          0.01 |          0.01 |          0.01 |
| LoRA rank / alpha       |       64 / 64 |       64 / 64 |       64 / 64 |       64 / 64 |
| Image resolution        |           448 |           336 |           336 |           336 |

### Training scripts

All scripts use the MMEB VQA subsets and reside in `script_train/`.

| Backbone              | Full CSM-VL                  | UG ablation                | CMI ablation                |
| --------------------- | ---------------------------- | -------------------------- | --------------------------- |
| FastVLM-0.5B          | `fastvlm_cms_full_vqa.sh`    | `fastvlm_cms_ug_vqa.sh`    | `fastvlm_cms_cmi_vqa.sh`    |
| Qwen3-VL-Embedding-2B | `qwen3vl_2b_cms_full_vqa.sh` | `qwen3vl_2b_cms_ug_vqa.sh` | `qwen3vl_2b_cms_cmi_vqa.sh` |
| Qwen3-VL-Embedding-8B | `qwen3vl_8b_cms_full_vqa.sh` | `qwen3vl_8b_cms_ug_vqa.sh` | `qwen3vl_8b_cms_cmi_vqa.sh` |

Classification runs use the ten MMEB classification subsets
(`ImageNet_1K`, `HatefulMemes`, `SUN397`, `N24News`, `VOC2007`, `Place365`,
`ImageNet-A`, `ImageNet-R`, `ObjectNet`, and `Country211`):

| Backbone              | Full CSM-VL                  | UG ablation                | CMI ablation                |
| --------------------- | ---------------------------- | -------------------------- | --------------------------- |
| FastVLM-0.5B          | `fastvlm_cms_full_cls.sh`    | `fastvlm_cms_ug_cls.sh`    | `fastvlm_cms_cmi_cls.sh`    |
| Qwen3-VL-Embedding-2B | `qwen3vl_2b_cms_full_cls.sh` | `qwen3vl_2b_cms_ug_cls.sh` | `qwen3vl_2b_cms_cmi_cls.sh` |
| Qwen3-VL-Embedding-8B | `qwen3vl_8b_cms_full_cls.sh` | `qwen3vl_8b_cms_ug_cls.sh` | `qwen3vl_8b_cms_cmi_cls.sh` |

For example, train the full 8B model with:

```bash
bash script_train/qwen3vl_8b_cms_full_vqa.sh
```

The 8B scripts follow the 2B configuration (batch size 16 and a `1e-4`
learning rate). Reduce the batch size only if required by available hardware.

Existing baseline scripts for fixed MRL and ESE remain available in the same
directory.

## Evaluation

To evaluate a checkpoint on an MMEB task such as `MSCOCO_i2t`, configure the
checkpoint path in the supplied evaluation script and run:

```bash
bash eval.sh
```

## Acknowledgement

This codebase adapts components from
[VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec) and
[B3](https://github.com/raghavlite/B3).


