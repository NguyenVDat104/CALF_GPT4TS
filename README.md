---
title: CALF Time Series Forecasting
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Taming Pre-trained LLMs for Generalized Time Series Forecasting through Cross-modal Knowledge Distillation

[![](http://img.shields.io/badge/cs.LG-arXiv%3A2403.07300-B31B1B.svg)](https://arxiv.org/abs/2403.07300)

---

## 🚀 [NEW UPDATE] DINO-V3 Vision Backbone & Encoder Integration

This repository has been updated to integrate **Meta's DINO-V3** Vision Foundation Model into the CALF framework:
1. **Backbone (`--backbone dinov3`)**: Replaces the GPT-2 LLM backbone with DINO-V3 Vision Transformer blocks, extracting multi-layer intermediate representations for Cross-Modal Knowledge Distillation.
2. **Input Encoder (`--encoder dinov3`)**: Replaces the default `Encoder_PCA` with a DINO-V3 ViT Patch/Feature Encoder, projecting 1D time series channels into visual representation tokens and aligning them with word token embeddings via cross-attention.
3. **End-to-End DINO-V3 (`--backbone dinov3 --encoder dinov3`)**: Both encoder and backbone operate using the DINO-V3 Vision Transformer architecture.

### 📌 Summary of Updated and Added Files
* **`models/dinov3_arch.py`** *(NEW FILE)*:
  * `DINOv3Backbone`: ViT backbone adapter supporting Hugging Face checkpoints (`facebook/dinov3-*`) with an automatic graceful fallback to a standalone pure PyTorch DINO-V3 ViT (with QK-Normalization, LayerScale, and intermediate states extraction).
  * `Encoder_DINOv3`: Visual feature tokenizer and cross-modal attention module for time series.
  * `CustomDINOv3ViT`, `DINOv3Block`, `DINOv3Attention`, `DINOv3LayerScale`: Pure PyTorch implementations of core DINO-V3 building blocks.
* **`models/GPT4TS.py` & `models/GPT4TS_ETT.py`** *(UPDATED)*:
  * Added configurable support for `--backbone` (`gpt2` or `dinov3`) and `--encoder` (`pca`, `dinov3`, or `adaptive`).
  * Ensured seamless compatibility with `DistillationLoss` (loss over intermediate features and output logits).
* **`run.py`** *(UPDATED)*:
  * Added CLI parameters: `--backbone`, `--encoder`, `--dinov3_model_name`, `--dinov3_layers`, `--dinov3_pretrained`.
  * Updated experiment setting string to automatically record backbone and encoder configurations in checkpoint paths.
* **`scripts/long_term_forecasting/ETTh_DINOv3.sh`** *(NEW FILE)*:
  * Ready-to-run automation script benchmarking DINO-V3 Backbone, DINO-V3 Encoder, and End-to-End configurations.

### ⚙️ New Command-Line Options
| Argument | Type | Default | Options / Description |
| :--- | :--- | :--- | :--- |
| `--backbone` | `str` | `gpt2` | `gpt2` (original) or `dinov3` (DINO-V3 Vision Transformer backbone) |
| `--encoder` | `str` | `pca` | `pca` (original), `dinov3` (DINO-V3 ViT encoder), or `adaptive` |
| `--dinov3_model_name` | `str` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Hugging Face checkpoint name or local directory path |
| `--dinov3_layers` | `int` | `6` | Number of ViT layers to utilize in DINO-V3 |
| `--dinov3_pretrained` | `int` | `1` | `1`: Attempt loading HF pretrained weights (auto-fallback to offline PyTorch ViT); `0`: Pure PyTorch ViT initialized from scratch |

### 💻 Usage Examples

**1. Run CALF with DINO-V3 as Backbone:**
```bash
python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh1_96_96_dinov3_bb \
    --data ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --backbone dinov3 \
    --encoder pca \
    --dinov3_layers 6 \
    --log_fine_name dinov3_results.txt
```

**2. Run CALF with DINO-V3 as Encoder:**
```bash
python run.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh1_96_96_dinov3_enc \
    --data ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --backbone gpt2 \
    --encoder dinov3 \
    --log_fine_name dinov3_results.txt
```

**3. Run full automated ablation batch script:**
```bash
sh scripts/long_term_forecasting/ETTh_DINOv3.sh
```

---


## Prerequisites
Before proceeding, ensure Python 3.9 is installed. Install the required dependencies with the following command:

```
pip install -r requirements.txt
```

## Dataset Preparation

### Long-term Forecasting
Acquire datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Organize them in the `./datasets` directory as shown below:

```
datasets
├── electricity
│   └── electricity.csv
├── ETT-small
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── traffic
│   └── traffic.csv
└── weather
    └── weather.csv
```

### Short-term Forecasting
For short-term forecasting, download the M4 datasets from [Time-Series-Library](https://drive.google.com/drive/folders/15zio96o3NK4XOoR5L88oaWcJDVOiqQo9). Place the `m4` folder within `./datasets`.

## Preparing Word Token Embeddings

Execute the command below to extract principal components from the word token embeddings:

```
python pca.py
```

These embeddings will be saved in `./wte_pca_500.pt`.

## Model Training

Training scripts are located in the `./scripts` folder. For instance, to train the LLaTA model on the ETTh2 dataset for long-term forecasting, execute:

```
sh scripts/long_term_forecasting/ETTh2.sh
```

For short-term forecasting, use:

```
sh scripts/short_term_forecasting/m4.sh
```

Post-Training:

- Trained models will be saved in `./checkpoints`.
- Numerical results are available in `.npy` format under `./results`.
- Detailed summaries of performance metrics can be found in `./results_{task_name}.txt`.

## Citation
If this repository contributes to your research, please consider citing our work:

```
@article{liu2024taming,
      title={Taming Pre-trained LLMs for Generalised Time Series Forecasting via Cross-modal Knowledge Distillation}, 
      author={Liu, Peiyuan and Guo, Hang and Dai, Tao and Li, Naiqi and Bao, Jigang and Ren, Xudong and Jiang, Yong and Xia, Shu-Tao},
      journal={arXiv preprint arXiv:2403.07300},
      year={2024},
      arxiv={2403.07300}
}
```

## Acknowledgements

Our gratitude extends to the authors of the following repositories for their foundational model implementations:

- [Autoformer](https://github.com/thuml/Autoformer)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [Crossformer](https://github.com/Thinklab-SJTU/Crossformer)
- [One-Fits-All](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)

## Contact Us
For inquiries or further assistance, contact us at [lpy23@mails.tsinghua.edu.cn](mailto:lpy23@mails.tsinghua.edu.cn) or open an issue on this repository.
