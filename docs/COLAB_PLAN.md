# Google Colab Plan (15-16GB VRAM)

This guide is for continuing DTR dataset generation from a partial checkpoint, then training and uploading LoRA weights.

## 1) Runtime + Clone
- In Colab: `Runtime -> Change runtime type -> GPU`
- Use this in a cell:

```bash
%cd /content
!git clone https://github.com/seshuthota/DTR-explorer.git
%cd /content/DTR-explorer
!bash scripts/colab/00_setup.sh
```

## 2) Restore Your Downloaded Partial Data
Upload your local files into `/content/DTR-explorer/data/`:
- `dtr_filtered_sft.jsonl`
- `dtr_candidates_debug.csv`
- `dtr_filtered_sft.jsonl.state.json`

Then verify:

```bash
!ls -lah data
!wc -l data/dtr_filtered_sft.jsonl data/dtr_candidates_debug.csv
```

## 3) Resume Generation (Colab-safe defaults)
Defaults are auto-profiled by VRAM (`AUTO_PROFILE=1`):
- strict filter: `require_correct=1`, `require_boxed=1`, `exclude_truncated=1`
- speed profile sets `SAMPLE_BATCH_SIZE`, `MAX_NEW_TOKENS`, `DTR_MAX_TOKENS`
- `RESUME=1`

```bash
!QUESTIONS=200 SAMPLES_PER_Q=16 RESUME=1 \
  bash scripts/colab/10_generate_dataset.sh
```

Manual override example (if you want to force larger parallelism):
```bash
!AUTO_PROFILE=0 SAMPLE_BATCH_SIZE=16 MAX_NEW_TOKENS=448 DTR_MAX_TOKENS=224 \
  QUESTIONS=200 SAMPLES_PER_Q=16 RESUME=1 \
  bash scripts/colab/10_generate_dataset.sh
```

## 4) Train LoRA (Colab-safe defaults)
Defaults are auto-profiled by VRAM (`AUTO_PROFILE=1`) and disable in-train eval for speed.

```bash
!DATASET_PATH=data/dtr_filtered_sft.jsonl \
  OUT_DIR=models/dtr-tuned-1.2b-colab \
  bash scripts/colab/20_train_sft.sh
```

Manual override example:
```bash
!AUTO_PROFILE=0 MAX_SEQ_LEN=768 MICRO_BATCH=2 GRAD_ACCUM=8 EPOCHS=1 \
  DATASET_PATH=data/dtr_filtered_sft.jsonl \
  OUT_DIR=models/dtr-tuned-1.2b-colab \
  bash scripts/colab/20_train_sft.sh
```

## 5) Upload Weights to Hugging Face
```bash
import os
os.environ["HF_TOKEN"] = "hf_xxx"
```

```bash
!HF_REPO_ID=username/dtr-tuned-1.2b-colab-lora \
  LOCAL_DIR=models/dtr-tuned-1.2b-colab \
  bash scripts/colab/40_upload_hf.sh
```

## 6) Optional Quick Eval
```bash
!MODEL_ID=models/dtr-tuned-1.2b-colab bash scripts/colab/30_eval.sh
```

Outputs:
- `outputs/think_n_posttrain_colab.csv`
- `outputs/len_vs_dtr_posttrain_colab.csv`
- `outputs/len_vs_dtr_posttrain_colab.png`
