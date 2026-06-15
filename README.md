# orbit_warsv2

`orbit_warsv2` is a board-level behavioral cloning workflow for Orbit Wars. It builds datasets from replay JSON files, trains a BC policy, evaluates checkpoints, and exports a runnable Python submission file.

## Setup

Linux/macOS:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Windows PowerShell:

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -e .
```

Run tests:

```bash
python -m pytest
```

## Replay input

Put replay JSON files in `./replays/`:

```text
replays/
  episode-0001-replay.json
  episode-0002-replay.json
  ...
```

## Build dataset

Small smoke-test build:

```bash
python -m orbit_board_bc_data.cli build \
  --replay-dir ./replays \
  --out-dir ./orbit_dataset_work/board_bc_small \
  --player-filter winner \
  --valid-ratio 0.1 \
  --seed 13 \
  --max-files 10 \
  --workers 1
```

Full build:

```bash
python -m orbit_board_bc_data.cli build \
  --replay-dir ./replays \
  --out-dir ./orbit_dataset_work/board_bc \
  --player-filter winner \
  --valid-ratio 0.1 \
  --seed 13 \
  --max-planets 40 \
  --max-fleets 256 \
  --max-actions-per-turn 32 \
  --workers 8 \
  --worker-output shard
```

Append up to 100 new replay files:

```bash
python -m orbit_board_bc_data.cli build \
  --replay-dir ./replays \
  --out-dir ./orbit_dataset_work/board_bc \
  --append \
  --max-files 100 \
  --workers 8 \
  --worker-output shard
```

Build arguments:

| Argument | Meaning |
| --- | --- |
| `--replay-dir` | Directory containing replay `.json` files. |
| `--out-dir` | Output dataset directory. |
| `--player-filter` | Rows to extract: `winner`, `top2`, or `all`. |
| `--valid-ratio` | Fraction of episodes used for validation. |
| `--seed` | Seed for deterministic train/validation split. |
| `--max-files` | Maximum replay JSON files to process. Omit it for a full build. With `--append`, existing episodes are skipped first, then this limit is applied. |
| `--max-planets` | Maximum planet tokens per sample. |
| `--max-fleets` | Maximum fleet tokens per sample. |
| `--max-actions-per-turn` | Maximum action labels per turn. |
| `--workers` | Number of replay worker processes. |
| `--worker-output` | Worker write mode: `shard` or `parent`. |
| `--append` | Add only new episodes to an existing compatible dataset. |

## Validate dataset

```bash
python -m orbit_board_bc_data.cli validate \
  --dataset ./orbit_dataset_work/board_bc \
  --unmatched-threshold 0.01 \
  --ambiguous-threshold 0.01
```

Validation arguments:

| Argument | Meaning |
| --- | --- |
| `--dataset` | Dataset directory to validate. |
| `--unmatched-threshold` | Maximum allowed unmatched-label rate. |
| `--ambiguous-threshold` | Maximum allowed ambiguous-label rate. |

## Feature probe

```bash
python -m orbit_board_bc_data.cli feature-probe \
  --dataset ./orbit_dataset_work/board_bc \
  --out-dir ./orbit_dataset_work/feature_probe
```

Feature-probe arguments:

| Argument | Meaning |
| --- | --- |
| `--dataset` | Dataset directory to inspect. |
| `--out-dir` | Directory where `feature_probe.json` is written. |

## Train BC policy

```bash
python -m orbit_board_bc_train.cli train \
  --dataset ./orbit_dataset_work/board_bc \
  --out-dir ./bc_runs/board_bc_v1 \
  --hidden-dim 192 \
  --encoder-layers 4 \
  --decoder-layers 2 \
  --heads 6 \
  --dropout 0.05 \
  --batch-size 128 \
  --epochs 20 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --noop-stop-weight 0.35 \
  --device auto \
  --resume ./bc_runs/board_bc_v1/last.pt
```

Training arguments:

| Argument | Meaning |
| --- | --- |
| `--dataset` | Dataset root containing `train/` and `valid/`. |
| `--out-dir` | Training output directory. |
| `--hidden-dim` | Model hidden size. |
| `--encoder-layers` | Number of encoder layers. |
| `--decoder-layers` | Number of decoder layers. |
| `--heads` | Number of attention heads. |
| `--dropout` | Dropout rate. |
| `--batch-size` | Training batch size. |
| `--epochs` | Number of training epochs. |
| `--lr` | Learning rate. |
| `--weight-decay` | Weight decay. |
| `--grad-clip` | Gradient clipping norm. |
| `--noop-stop-weight` | Noop stop loss weight. |
| `--device` | Training device, usually `auto`. |
| `--resume` | Optional training checkpoint to resume from, usually `last.pt`. `--epochs` remains the target total epoch count, not the number of extra epochs. |

`last.pt` stores the most recent completed epoch plus optimizer state. To continue a run interrupted after epoch 7 and train through epoch 20, rerun the same command with `--epochs 20 --resume ./bc_runs/board_bc_v1/last.pt`.

## Evaluate checkpoint

```bash
python -m orbit_board_bc_train.cli eval \
  --dataset ./orbit_dataset_work/board_bc \
  --checkpoint ./bc_runs/board_bc_v1/best/checkpoint.pt \
  --batch-size 128 \
  --device auto
```

Evaluation arguments:

| Argument | Meaning |
| --- | --- |
| `--dataset` | Dataset root used for evaluation. |
| `--checkpoint` | Checkpoint file to evaluate. |
| `--batch-size` | Evaluation batch size. |
| `--device` | Evaluation device. |

## Export submission file

```bash
python -m orbit_board_bc_train.cli export-agent \
  --checkpoint ./bc_runs/board_bc_v1/best/checkpoint.pt \
  --out ./submission/main.py
```

Export arguments:

| Argument | Meaning |
| --- | --- |
| `--checkpoint` | Trained checkpoint to export. |
| `--out` | Output Python file. |

## Workflow

```text
replays -> build dataset -> validate dataset -> train BC -> evaluate checkpoint -> export submission file
```
