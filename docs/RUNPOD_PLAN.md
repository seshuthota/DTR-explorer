# RunPod Training Plan (DTR Explorer)

## Goal
Train a DTR-informed SFT model using QLoRA on RunPod, then validate with Think@n and correlation tests.

## Recommended Pod Specs
- Preferred: 24GB+ VRAM (L4 / RTX 4090 / A5000 class)
- Minimum viable: 12GB with slower settings
- 6GB can train only with very conservative settings and is not recommended for the full data generation + training loop.

## 1) Start Pod and Clone Repo
```bash
cd /workspace
git clone <your-repo-url> dtr-explorer
cd dtr-explorer
bash scripts/runpod/00_setup.sh
source .venv-runpod/bin/activate
```

Optional (for gated/private HF models):
```bash
export HF_TOKEN=hf_xxx
huggingface-cli login --token "$HF_TOKEN"
```

## 2) Generate DTR-Filtered SFT Data
```bash
bash scripts/runpod/10_generate_dataset.sh
```
Default settings (editable via env vars):
- `QUESTIONS=200`
- `SAMPLES_PER_Q=16`
- `MAX_NEW_TOKENS=320`
- `MIN_DTR=0.32`
- `KEEP_PER_Q=4`
- filters: `--require-correct --fallback-best-correct`

Output:
- `data/dtr_filtered_sft.jsonl`
- `data/dtr_candidates_debug.csv`

## 3) Train LoRA Adapters (QLoRA)
```bash
bash scripts/runpod/20_train_sft.sh
```
Default training profile:
- base model: `DavidAU/LFM2.5-1.2B-...-DISTILL`
- `max_seq_len=1024`, `epochs=2`, `lr=2e-5`
- micro batch `1`, grad accum `16`
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`

Output:
- `models/dtr-tuned-1.2b-v1` (LoRA adapter)

## 4) Post-Train Validation
```bash
MODEL_ID=models/dtr-tuned-1.2b-v1 bash scripts/runpod/30_eval.sh
```
Outputs:
- `outputs/think_n_posttrain.csv`
- `outputs/len_vs_dtr_posttrain.csv`
- `outputs/len_vs_dtr_posttrain.png`

## 5) Suggested Profiles

### 24GB profile (recommended)
```bash
QUESTIONS=250 SAMPLES_PER_Q=16 MAX_NEW_TOKENS=320 bash scripts/runpod/10_generate_dataset.sh
MAX_SEQ_LEN=1024 MICRO_BATCH=2 GRAD_ACCUM=8 bash scripts/runpod/20_train_sft.sh
```

### 6GB fallback profile (slow / fragile)
```bash
QUESTIONS=80 SAMPLES_PER_Q=8 MAX_NEW_TOKENS=192 MIN_DTR=0.30 KEEP_PER_Q=2 bash scripts/runpod/10_generate_dataset.sh
MAX_SEQ_LEN=512 MICRO_BATCH=1 GRAD_ACCUM=32 LORA_R=8 LORA_ALPHA=16 bash scripts/runpod/20_train_sft.sh
```

## 6) Operational Notes
- Persist `/workspace` or attach a network volume before long runs.
- If OOM occurs, reduce in order: `MAX_SEQ_LEN`, `MICRO_BATCH`, `LORA_R`, then increase `GRAD_ACCUM`.
- Keep `data/dtr_candidates_debug.csv` for threshold analysis before retraining.
