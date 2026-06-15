# Orbit Wars V2

Short project description: Orbit Wars V2 is a training workflow for building an Orbit Wars agent from replay data. The pipeline prepares replay-derived datasets, trains a behavioral cloning policy, and can optionally fine-tune that policy with PPO.

## 1. Environment setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

On Windows PowerShell:

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -e .
```

Arguments:

| Command part | Meaning |
| --- | --- |
| `uv venv --python 3.12` | Creates a local virtual environment with Python 3.12. |
| `source .venv/bin/activate` / `.venv\Scripts\Activate.ps1` | Activates the virtual environment. |
| `uv pip install -e .` | Installs this project in editable mode so local code changes are used immediately. |

## 2. Prepare replay files

Put replay JSON files under:

```text
./replays/
```

Expected layout:

```text
replays/
  episode-0001-replay.json
  episode-0002-replay.json
  ...
```

Optional KaggleHub download pattern:

```bash
python - <<'PY'
import kagglehub

path = kagglehub.dataset_download(
    "kaggle/orbit-wars-episodes-2026-06-14",
    path="./replays",
)
print("Replay files downloaded to:", path)
PY
```

Arguments:

| Argument | Meaning |
| --- | --- |
| `kaggle/orbit-wars-episodes-2026-06-14` | Kaggle dataset slug. Replace it if using another replay dataset. |
| `path="./replays"` | Local folder where replay files should be stored. |

## 3. Build the dataset

Small CPU/default run:

```bash
python -m orbit_training_prep.dataset_builder \
  --replay ./replays \
  --out-dir ./orbit_dataset_work/combined \
  --horizon 160 \
  --max-file 10
```

CUDA/parallel run:

```bash
python -m orbit_training_prep.dataset_builder \
  --replay ./replays \
  --out-dir ./orbit_dataset_work/combined_cuda \
  --horizon 160 \
  --device cuda \
  --batch-size 256 \
  --workers 1 \
  --max-file 100
```

For full dataset generation, remove `--max-file`.

Arguments:

| Argument | Meaning |
| --- | --- |
| `--replay` | Input replay directory. |
| `--out-dir` | Output dataset directory. |
| `--horizon` | Future-step horizon used when deriving training labels from replay outcomes. |
| `--device` | Compute device for supported preprocessing paths. Use `cuda` when GPU support is available. |
| `--batch-size` | Number of replay/source-turn items processed per batch. |
| `--workers` | Number of worker processes used for preprocessing. |
| `--max-file` | Maximum number of replay files to process. Use this for smoke tests or limited dataset builds. Remove it for the full dataset. |

## 4. Train the BC policy

```bash
python -m orbit_bc_training.train_bc_policy \
  --train_dir ./orbit_dataset_work/combined_split/train \
  --valid_dir ./orbit_dataset_work/combined_split/valid \
  --out_dir ./bc_checkpoints/run_001 \
  --batch_size 512 \
  --epochs 20 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --hidden_size 128 \
  --num_layers 2 \
  --num_heads 4 \
  --mlp_size 256 \
  --dropout 0.0 \
  --seed 42 \
  --device auto \
  --num_workers 0
```

Arguments:

| Argument | Meaning |
| --- | --- |
| `--train_dir` | Training split directory. |
| `--valid_dir` | Validation split directory. |
| `--out_dir` | Checkpoint/output directory. |
| `--batch_size` | Training batch size. |
| `--epochs` | Number of full passes over the training data. |
| `--lr` | Learning rate. |
| `--weight_decay` | L2 regularization strength. |
| `--grad_clip` | Maximum gradient norm before clipping. |
| `--hidden_size` | Transformer/model hidden dimension. |
| `--num_layers` | Number of model layers. |
| `--num_heads` | Number of attention heads. |
| `--mlp_size` | Feed-forward layer size. |
| `--dropout` | Dropout rate. Use `0.0` for deterministic full-data training. |
| `--seed` | Random seed for reproducibility. |
| `--device` | Training device. `auto` selects CUDA when available, otherwise CPU. |
| `--num_workers` | DataLoader worker count. Use `0` when multiprocessing causes memory or platform issues. |

Best checkpoint path after training:

```text
./bc_checkpoints/run_001/best/checkpoint.pt
```

## 5. Optional PPO fine-tuning

```bash
python -m orbit_ppo_jax.train \
  --bc_checkpoint ./bc_checkpoints/run_001/best/checkpoint.pt \
  --out_dir ./ppo_runs/jax_ppo_4p \
  --players 4 \
  --envs 80 \
  --enable_comets \
  --rollout_steps 32 \
  --episode_steps 500 \
  --updates 5 \
  --pfsp_max_policy_slots 8 \
  --pfsp_matrix_games 0 \
  --eval_games 0 \
  --source_cap 3 \
  --precision float16 \
  --no_remat_policy_eval
```

Arguments:

| Argument | Meaning |
| --- | --- |
| `--bc_checkpoint` | Starting BC checkpoint used to initialize PPO. |
| `--out_dir` | PPO run output directory. |
| `--players` | Number of players in the environment. |
| `--envs` | Number of parallel rollout environments. |
| `--enable_comets` | Enables comet mechanics in the environment. |
| `--rollout_steps` | Number of environment steps collected per PPO rollout. |
| `--episode_steps` | Maximum steps per episode. |
| `--updates` | Number of PPO update iterations. |
| `--pfsp_max_policy_slots` | Maximum opponent-policy slots for PFSP. |
| `--pfsp_matrix_games` | Number of games used to update the PFSP matchup matrix. `0` disables matrix evaluation. |
| `--eval_games` | Number of evaluation games per run. `0` disables evaluation. |
| `--source_cap` | Maximum number of source planets/actions considered per decision. |
| `--precision` | Numeric precision used by the JAX PPO run. |
| `--no_remat_policy_eval` | Disables rematerialization during policy evaluation to reduce complexity/debug issues. |

## Common outputs

| Path | Meaning |
| --- | --- |
| `./orbit_dataset_work/` | Generated datasets and intermediate files. |
| `./bc_checkpoints/` | Behavioral cloning checkpoints. |
| `./ppo_runs/` | PPO fine-tuning outputs. |

## Recommended workflow

```text
replays -> dataset_builder -> BC training -> BC checkpoint -> optional PPO fine-tuning
```
