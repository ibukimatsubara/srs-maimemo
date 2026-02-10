# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SRS Benchmark evaluates the predictive accuracy of 30+ spaced repetition algorithms on ~727 million Anki flashcard reviews (10k users). Algorithms predict the probability of recall for each review, and are scored on Log Loss, RMSE(bins), and AUC.

## Commands

### Run a benchmark
```bash
uv run script.py --algo FSRS-6 --short
```

Key flags: `--algo <name>`, `--short` (include same-day reviews), `--secs` (use seconds instead of days), `--duration` (review duration feature, LSTM only), `--default` (skip training, use default params), `--S0` (optimize only initial stability), `--two_buttons` (merge Hard/Easy into Good), `--recency` (recency weighting), `--partitions [none|deck|preset]`, `--processes N` (default 8), `--gpus <ids|all>`, `--dev` (local development).

### Lint and type check
```bash
uvx ruff check .
uvx mypy .
```

### Install dependencies
```bash
uv sync
```

## Architecture

### Entry point: `script.py`
The main orchestrator. Contains `Trainer` class for neural network training, TimeSeriesSplit cross-validation, and multiprocessing dispatch. Processes each user independently via `ProcessPoolExecutor`.

### Algorithm flow
1. `config.py` — Parses CLI args into a `Config` object. Defines `ModelName` literal type for all supported algorithms.
2. `data_loader.py` — `UserDataLoader` loads Parquet revlogs per user, applies feature engineering via the configured feature engineer.
3. `features/factory.py` — Registry mapping `ModelName` → `BaseFeatureEngineer` subclass. Strategy pattern: each algorithm family has its own feature engineer.
4. `models/model_factory.py` — Registry mapping `ModelName` → model class. Creates and initializes models. Some models (SM2, Ebisu-v2, AVG, MOVING-AVG, FSRS-rs, FSRS-6-one-step, RMSE-BINS-EXPLOIT) have special processing flows in `model_processors.py` and are not in this registry.
5. `evaluate.py` / `utils.py` — Evaluation pipeline: computes Log Loss, RMSE(bins), AUC per user; aggregates results.

### Key patterns
- **Two registries**: `FEATURE_ENGINEER_REGISTRY` in `features/factory.py` and `MODEL_REGISTRY` in `models/model_factory.py`. Both keyed by `ModelName`. When adding a new algorithm, register in both.
- **Special-case models**: Several models bypass the standard factory and are handled in `model_processors.py` (untrainable models, Rust wrappers, baselines).
- **Reptile meta-learning** (`reptile_trainer.py`): Used for LSTM. Pretrain on 100 users, then fine-tune per user.
- **RWKV** (`rwkv/`): Separate pipeline with custom CUDA kernels. Trained on 5k users, evaluated on another 5k. Excluded from mypy.

### Data
- Dataset: `open-spaced-repetition/anki-revlogs-10k` on Hugging Face
- Storage: LMDB for shared metric computation across workers
- Results: JSON Lines in `result/`, per-user TSVs in `evaluation/`

## Code Quality

- **Linter**: Ruff (`ruff.toml` — ignores F405, E741, F403, F401, E402, E712)
- **Type checker**: mypy (`mypy.ini` — excludes `rwkv/`)
- **Python**: 3.12+
- **Package manager**: uv
