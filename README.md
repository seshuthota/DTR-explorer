# DTR Explorer

A replication and extension of the **"Think Deep, Not Just Long"** paper — measuring reasoning depth in language models via internal prediction convergence (Deep-Thinking Ratio).

## Project Structure

```
DTR Explorer/
├── dtr/                        # Core library
│   ├── model.py                # HuggingFace model wrapper with hidden state extraction
│   └── calculator.py           # DTR metric: JSD settling depth + Top-K agreement
├── experiments/                # Reproducible experiment scripts
│   ├── threshold_sweep.py      # Sweep g threshold, print settling histograms
│   ├── temperature_sweep.py    # Temperature robustness (T=0.0/0.4/0.8)
│   ├── cot_vs_direct.py        # Control: CoT vs Direct-answer prompting
│   ├── token_type_analysis.py  # Per-token-type settling depth breakdown
│   ├── think_n.py              # Think@n (prefix DTR ranking + early pruning)
│   ├── length_vs_dtr_correlation.py # High-sample length-vs-DTR correlation + plots
│   ├── generate_dtr_dataset.py # Build DTR-filtered SFT dataset
│   ├── dtr_accuracy_correlation.py # DTR vs correctness correlation + quintile bins
│   └── jsd_diagnostic.py       # Raw JSD-per-layer diagnostic
├── run_experiment.py           # Quick GSM8K benchmark runner
├── WALKTHROUGH.md              # Full replication writeup with results
├── paper.md                    # Extracted paper text for reference
└── 2602.13517v1.pdf            # Original paper PDF
```

## Quick Start

```bash
# Activate the environment
conda activate rl

# Run a quick GSM8K benchmark
python run_experiment.py

# Run the threshold calibration sweep
python experiments/threshold_sweep.py

# Run the temperature robustness experiment
python experiments/temperature_sweep.py

# Run the CoT vs Direct-answer control
python experiments/cot_vs_direct.py

# Run sample-level DTR vs accuracy correlation
python experiments/dtr_accuracy_correlation.py --questions 20 --samples-per-question 8

# Run Think@n vs full self-consistency
python experiments/think_n.py --questions 30 --n 24 --prefix-tokens 50 --keep-ratio 0.5

# Run high-sample length-vs-DTR correlation (with plot)
python experiments/length_vs_dtr_correlation.py --questions 15 --samples-per-question 50 --plot-out outputs/len_vs_dtr.png

# Build DTR-filtered SFT data (recommended first training step)
python experiments/generate_dtr_dataset.py \
  --questions 200 \
  --samples-per-q 16 \
  --min-dtr 0.32 \
  --keep-per-q 4 \
  --require-correct \
  --fallback-best-correct \
  --output data/dtr_filtered_sft.jsonl

# Keep close to full 3,200 rows (200x16), less strict filtering
python experiments/generate_dtr_dataset.py \
  --questions 200 \
  --samples-per-q 16 \
  --keep-per-q 16 \
  --min-dtr 0.0 \
  --output data/dtr_filtered_sft_3200.jsonl
```

## Key Findings

| Experiment | Result |
|------------|--------|
| Calibrated threshold | g=0.60 (vs paper's 0.50) needed for 1.2B model |
| Math vs Easy DTR | Math consistently +3–6 pp higher across all conditions |
| Temperature robustness | DTR stable across T=0.0/0.4/0.8 |
| CoT vs Direct | DTR driven by task difficulty, not prompt verbosity |

See [WALKTHROUGH.md](WALKTHROUGH.md) for the full writeup.

## Model

Currently uses `DavidAU/LFM2.5-1.2B-Thinking-Claude-4.6-Opus-Heretic-Uncensored-DISTILL` (auto-downloaded to `models/`).

## RunPod Pipeline

RunPod-ready files are included for end-to-end dataset generation + QLoRA training:

- Setup: `scripts/runpod/00_setup.sh`
- DTR dataset generation: `scripts/runpod/10_generate_dataset.sh`
- SFT training (LoRA): `scripts/runpod/20_train_sft.sh`
- Upload artifacts to HF: `scripts/runpod/40_upload_hf.sh`
- Post-train evaluation: `scripts/runpod/30_eval.sh`
- Full execution plan: `docs/RUNPOD_PLAN.md`

Google Colab wrappers are also included (15-16GB VRAM friendly defaults):

- Setup: `scripts/colab/00_setup.sh`
- Restore dataset from HF: `scripts/colab/06_restore_data_from_hf.sh`
- DTR dataset generation (resume-safe): `scripts/colab/10_generate_dataset.sh`
- SFT training (LoRA): `scripts/colab/20_train_sft.sh`
- Upload artifacts to HF: `scripts/colab/40_upload_hf.sh`
- Upload dataset checkpoint to HF: `scripts/colab/41_upload_dataset_hf.sh`
- Post-train evaluation: `scripts/colab/30_eval.sh`
- Colab execution plan: `docs/COLAB_PLAN.md`

Kaggle wrappers are included for direct notebook execution:

- Setup: `scripts/kaggle/00_setup.sh`
- Restore partial data checkpoint: `scripts/kaggle/05_restore_data.sh`
- Restore dataset from HF: `scripts/kaggle/06_restore_data_from_hf.sh`
- DTR dataset generation (resume-safe): `scripts/kaggle/10_generate_dataset.sh`
- SFT training (LoRA): `scripts/kaggle/20_train_sft.sh`
- Merge LoRA to standalone model: `scripts/kaggle/25_merge_lora.sh`
- Compare base vs merged: `scripts/kaggle/31_compare_base_vs_merged.sh`
- Upload artifacts to HF: `scripts/kaggle/40_upload_hf.sh`
- Upload dataset checkpoint to HF: `scripts/kaggle/41_upload_dataset_hf.sh`
- Post-train evaluation: `scripts/kaggle/30_eval.sh`
- Kaggle execution plan: `docs/KAGGLE_PLAN.md`

Notes on speed/quality defaults in Colab/Kaggle scripts:
- Auto-profile (`AUTO_PROFILE=1`) selects generation/training settings by detected VRAM.
- Dataset filtering defaults are strict for SFT quality: `require_correct=1`, `require_boxed=1`, `exclude_truncated=1`.
- Fast DTR option (`DTR_MAX_TOKENS`) is used to reduce scoring overhead while preserving ranking quality.

Quick start on RunPod:

```bash
bash scripts/runpod/00_setup.sh
source .venv-runpod/bin/activate
# 3,200 generated traces target (200 x 16)
QUESTIONS=200 SAMPLES_PER_Q=16 SAMPLE_BATCH_SIZE=2 RESUME=1 bash scripts/runpod/10_generate_dataset.sh
# Optional: keep near 3,200 rows for training
QUESTIONS=200 SAMPLES_PER_Q=16 KEEP_PER_Q=16 MIN_DTR=0.0 REQUIRE_CORRECT=0 FALLBACK_BEST_CORRECT=0 SAMPLE_BATCH_SIZE=2 RESUME=1 bash scripts/runpod/10_generate_dataset.sh
bash scripts/runpod/20_train_sft.sh
HF_TOKEN=hf_xxx HF_REPO_ID=username/dtr-tuned-1.2b-v1-lora bash scripts/runpod/40_upload_hf.sh
MODEL_ID=models/dtr-tuned-1.2b-v1 bash scripts/runpod/30_eval.sh
```
