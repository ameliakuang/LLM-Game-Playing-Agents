# Learning Game-Playing Agents with Generative Code Optimization

Code for *Learning Game-Playing Agents with Generative Code Optimization* (ICML 2025 PRAL Workshop). We use Trace LLM optimizers (OptoPrime) to optimize Python policies to play Atari games via object-centric representations (OC_Atari).

Paper: https://openreview.net/forum?id=ZM65X3NoTd

## Supported Games

Asterix, Breakout, Enduro, Freeway, Pong, Q*bert, Seaquest, Space Invaders

## Setup

### 1. Install dependencies

```bash
bash install.sh
```

This will:
- Install [uv](https://github.com/astral-sh/uv) if not already present
- Clone the [OC_Atari](https://github.com/ameliakuang/OC_Atari) library into `external/OC_Atari/`
- Install all Python dependencies via `uv sync`

### 2. Configure environment variables

Follow the [LLM API Setup of Trace](https://github.com/microsoft/Trace?tab=readme-ov-file#llm-api-setup) to use OptoPrime as the supported optimizer.

## Running Training

Each game has a corresponding training script. Run with:

```bash
uv run python <game>_training.py
```

For example:

```bash
uv run python asterix_training.py
uv run python breakout_training.py
uv run python pong_training.py
```

## Project Structure

```
├── *_training.py          # Training scripts (one per game)
├── trace_envs/            # Traced environment wrappers (one per game)
├── training_utils.py      # Shared training utilities
├── logging_util.py        # Logging configuration
├── plotting_game_perf.py  # Performance visualization
├── install.sh             # Setup script
├── pyproject.toml         # Dependencies (managed by uv)
├── external/OC_Atari/     # Object-centric Atari library
├── logs/                  # Training logs
└── trace_ckpt/            # Optimizer checkpoints
```
