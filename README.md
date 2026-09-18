# VLMEmbed: Conditional Multimodal Subspace Matryoshka Learning

This repository trains vision-language retrieval encoders with Matryoshka
representations. In addition to the existing fixed-prefix MRL and ESE losses,
it now includes **Conditional Multimodal Subspace MRL (CMS-MRL)**: a router
selects an input-specific order of embedding groups, so every prefix is nested
but its active dimensions can differ across examples.

## Setup and data

```bash
python -m venv vlm
source vlm/bin/activate
pip install -r requirements.txt
bash download_traindata.sh
bash download_traindata_2.sh
```

Evaluation images can optionally be downloaded from the
[`TIGER-Lab/MMEB-eval`](https://huggingface.co/datasets/TIGER-Lab/MMEB-eval)
dataset. If your Transformers installation has the known Qwen2-VL image
processor issue, run `python fix_lib.py`.

## CMS-MRL

The implementation partitions the embedding into equal-sized, disjoint groups.
For each query, a lightweight MLP router ranks the groups. Selecting the first
`k` entries of that ranking creates a strictly nested subspace at each budget.
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

The full objective is `L_base + cms_utility_weight * L_UG +
cms_cmi_weight * L_CMI`. The training log includes `cms_base_loss`,
`cms_ug_loss`, and `cms_cmi_loss` for analysis.

### Configuration

Use one of the following loss names with `--kd_loss_type`:

| Loss name | Enabled terms | Purpose |
| --- | --- | --- |
| `cms_mrl` | Base + UG + CMI | Full CMS-MRL model. |
| `cms_ug` | Base + UG | Utility-guided-routing ablation. |
| `cms_cmi` | Base + CMI | Interaction-regularization ablation. |

Important options are `--cms_num_groups` (default `8`),
`--cms_router_hidden_dim` (default `256`), `--cms_utility_temperature`,
`--cms_utility_weight`, and `--cms_cmi_weight`. The embedding dimension does
not need to be divisible by the group count; CMS-MRL pads the final group only
while computing the loss.

### Training scripts

All scripts use the MMEB VQA subsets and reside in `script_train/`.

| Backbone | Full CMS-MRL | UG ablation | CMI ablation |
| --- | --- | --- | --- |
| FastVLM-0.5B | `fastvlm_cms_full_vqa.sh` | `fastvlm_cms_ug_vqa.sh` | `fastvlm_cms_cmi_vqa.sh` |
| Qwen3-VL-Embedding-2B | `qwen3vl_2b_cms_full_vqa.sh` | `qwen3vl_2b_cms_ug_vqa.sh` | `qwen3vl_2b_cms_cmi_vqa.sh` |
| Qwen3-VL-Embedding-8B | `qwen3vl_8b_cms_full_vqa.sh` | `qwen3vl_8b_cms_ug_vqa.sh` | `qwen3vl_8b_cms_cmi_vqa.sh` |

For example, train the full 8B model with:

```bash
bash script_train/qwen3vl_8b_cms_full_vqa.sh
```

The 8B scripts use a batch size of 4 and a `5e-6` learning rate; adjust these
for the memory available on your hardware. Existing baseline scripts for fixed
MRL and ESE remain available in the same directory.

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
