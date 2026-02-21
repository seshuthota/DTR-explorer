# Repository Guidelines

## Project Structure & Module Organization
- `dtr/`: core implementation.
- `dtr/model.py`: Hugging Face wrapper for generation with intermediate hidden states.
- `dtr/calculator.py`: DTR computation (JSD-based settling depth, optional Top-K agreement).
- `experiments/`: reproducible analysis scripts (`threshold_sweep.py`, `temperature_sweep.py`, `cot_vs_direct.py`, `token_type_analysis.py`, `jsd_diagnostic.py`).
- `run_experiment.py`: quick GSM8K benchmark runner.
- `paper.md`, `WALKTHROUGH.md`, `2602.13517v1.pdf`: paper context and results narrative.
- `models/`: downloaded/cached model weights; do not edit manually.

## Build, Test, and Development Commands
- `conda activate finRL`: activate the expected environment.
- `python3 run_experiment.py`: run a quick end-to-end GSM8K + DTR pass.
- `python3 experiments/threshold_sweep.py`: sweep `g` threshold and inspect settling-depth histograms.
- `python3 experiments/temperature_sweep.py`: check robustness across temperatures.
- `python3 experiments/cot_vs_direct.py`: compare CoT vs direct-answer prompting.
- `python3 -m py_compile dtr/*.py experiments/*.py run_experiment.py`: fast syntax check before PRs.

## Coding Style & Naming Conventions
- Python style: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Keep modules focused: model plumbing in `dtr/model.py`, metric logic in `dtr/calculator.py`, experiment orchestration in `experiments/`.
- Prefer small, explicit functions and short comments that explain non-obvious reasoning.
- Use consistent prompt formatting and token slicing to avoid hidden-state/token misalignment bugs.

## Testing Guidelines
- No formal unit-test suite is present yet; use script-level validation.
- Minimum validation for changes:
  1. `py_compile` passes.
  2. Run at least one relevant experiment script.
  3. Record key outputs (DTR means, depth medians/IQR) in PR notes.
- For metric changes, test both easy-text and math prompts to verify directional behavior.

## Commit & Pull Request Guidelines
- Git history is unavailable in this workspace snapshot, so no local convention could be inferred.
- Recommended commit format: `type(scope): summary` (e.g., `fix(calculator): apply final norm before lm_head`).
- PRs should include:
  1. What changed and why.
  2. Exact commands run.
  3. Before/after metric snippets or tables.
  4. Any reproducibility caveats (model, seed, temperature, threshold).
