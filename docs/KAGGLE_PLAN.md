# Kaggle Plan (P100 / single GPU)

Use this guide to continue from your partial dataset checkpoint, finish generation, train LoRA, and upload weights.

## 1) Start Notebook
- Kaggle Notebook settings:
  - Accelerator: `GPU`
  - Internet: `ON` (needed for HF model download/upload)
- Prefer `P100` over `T4x2` for this repo as-is (current scripts are single-GPU).

## 2) Clone + Setup
```bash
%cd /kaggle/working
!git clone https://github.com/seshuthota/DTR-explorer.git
%cd /kaggle/working/DTR-explorer
!bash scripts/kaggle/00_setup.sh
```

## 3) Restore Partial Data Checkpoint
If you uploaded your previous files as a Kaggle dataset:
```bash
!SOURCE_DIR=/kaggle/input/<your-checkpoint-dataset-slug> \
  bash scripts/kaggle/05_restore_data.sh
```

If you uploaded a `.tar.gz` archive:
```bash
!SOURCE_ARCHIVE=/kaggle/input/<your-checkpoint-dataset-slug>/dtr_gen_checkpoint_partial.tar.gz \
  bash scripts/kaggle/05_restore_data.sh
```

Or restore directly from a HF Dataset repo:
```bash
import os
os.environ["HF_TOKEN"] = "hf_xxx"  # optional for private repo
```
```bash
!HF_REPO_ID=username/dtr-generated-data-v1 \
  bash scripts/kaggle/06_restore_data_from_hf.sh
```

Verify:
```bash
!ls -lah data
!wc -l data/dtr_filtered_sft.jsonl data/dtr_candidates_debug.csv
```

## 4) Resume Generation
Default Kaggle profile is auto-selected by VRAM (`AUTO_PROFILE=1`) and uses strict keep rules:
- `require_correct=1`
- `require_boxed=1`
- `exclude_truncated=1`
- `TEMPERATURE=0.5`, `TOP_K=20`

```bash
!QUESTIONS=200 SAMPLES_PER_Q=16 RESUME=1 \
  bash scripts/kaggle/10_generate_dataset.sh
```

Manual override example:
```bash
!AUTO_PROFILE=0 QUESTIONS=200 SAMPLES_PER_Q=16 RESUME=1 SAMPLE_BATCH_SIZE=16 MAX_NEW_TOKENS=448 DTR_MAX_TOKENS=224 \
  bash scripts/kaggle/10_generate_dataset.sh
```

## 5) Train LoRA
```bash
!DATASET_PATH=data/dtr_filtered_sft.jsonl \
  OUT_DIR=models/dtr-tuned-1.2b-kaggle \
  bash scripts/kaggle/20_train_sft.sh
```

Manual override example:
```bash
!AUTO_PROFILE=0 MAX_SEQ_LEN=768 MICRO_BATCH=2 GRAD_ACCUM=8 EPOCHS=1 \
  DATASET_PATH=data/dtr_filtered_sft.jsonl \
  OUT_DIR=models/dtr-tuned-1.2b-kaggle \
  bash scripts/kaggle/20_train_sft.sh
```

## 6) Merge LoRA and Compare vs Base
Merge uploaded adapter into standalone weights:
```bash
!ADAPTER_ID=CuriousDragon/dtr-tuned-1.2b-DTR-lora \
  OUT_DIR=models/dtr-tuned-1.2b-DTR-merged \
  bash scripts/kaggle/25_merge_lora.sh
```

Run before/after comparison:
```bash
!BASE_MODEL_ID=DavidAU/LFM2.5-1.2B-Thinking-Claude-4.6-Opus-Heretic-Uncensored-DISTILL \
  MERGED_MODEL_ID=models/dtr-tuned-1.2b-DTR-merged \
  QUESTIONS=12 N=16 MAX_NEW_TOKENS=384 \
  bash scripts/kaggle/31_compare_base_vs_merged.sh
```

## 7) Upload Weights to Hugging Face
```python
import os
os.environ["HF_TOKEN"] = "hf_xxx"
```

```bash
!HF_REPO_ID=username/dtr-tuned-1.2b-kaggle-lora \
  LOCAL_DIR=models/dtr-tuned-1.2b-kaggle \
  bash scripts/kaggle/40_upload_hf.sh
```

Upload generation dataset checkpoint (for resume later):
```bash
!HF_REPO_ID=username/dtr-generated-data-v1 \
  bash scripts/kaggle/41_upload_dataset_hf.sh
```

## 8) Optional Eval
```bash
!MODEL_ID=models/dtr-tuned-1.2b-kaggle bash scripts/kaggle/30_eval.sh
```
