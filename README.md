# VisualRL · Visual Deep Reinforcement Learning Control

NUS CEG5301 coursework portfolio combining Cliff Walking SARSA/Q-learning experiments and a visual DQN for simulated inverted-pendulum control.

## Implementation

- Four 42×42 grayscale frames and four historical angular velocities form the DQN observation.
- A three-layer CNN is fused with tiled angular-velocity features and maps to nine discrete torque values in [-2, 2].
- DQN uses epsilon-greedy exploration, uniform replay, a soft-updated target network and a masked Bellman target.
- Cliff Walking compares SARSA and Q-learning, including a decaying-epsilon experiment.

## Setup and run

```bash
conda env create -f environment.yml
conda activate visualrl
python scripts/smoke_check.py
python train.py --no_render --no_video
```

For a short integration check without overwriting retained model/curve files:

```bash
python train.py --steps 32 --learn_start 4 --learn_freq 4 --save_freq 1000 --no_render --no_video --artifact_dir work/smoke
```

For the full default run, remove `--steps 32`. Without `--no_render --no_video`, reaching the configured rolling-return threshold opens a visual demonstration and saves a video. Run from the repository root. GPU selection is automatic when the installed PyTorch build supports CUDA.

```bash
python cliff_walking.py
python visualize_results.py --game Pendulum
```

`cliff_walking.ipynb` retains the historical policy-evaluation outputs. Its interactive plotting uses an older Jupyter backend; `cliff_walking.py` is the standalone static-plot entry point.

`run_experiments.py` runs three parameter sets sequentially. The CLI wiring was repaired during packaging, so future commands now apply the requested values. The runner overwrites shared model and curve files: copy historical artifacts first, and archive each run if making a new benchmark. Old A/B/C plots do not validate the repaired settings retroactively.

## Retained results

| Task | Evidence |
|---|---|
| Visual Pendulum DQN | Final stored rolling mean **-198.41**, at 69 logging points × 1,000 steps |
| DQN episode log | Last 100 completed episodes average **-198.41396088**, independently consistent with the stored curve |
| Cliff Walking | 100 training runs × 500 episodes per method; greedy evaluation of the final run's policy over 200 episodes gives Q-learning **-7** and SARSA **-9** |

`data/model/` contains retained prediction and target weights; `data/plots/` contains historical plots and curve data; `artifacts/` contains the retained episode CSV and video. The -200 threshold is a coursework demonstration trigger, not a rigorous convergence certificate. The DQN number is a rolling training return, not held-out multi-seed performance.

The retained checkpoints use **DQN_N_TILE=15**, inferred from their layer dimensions; the default for a new training run is 20. Set that environment variable to 15 before loading historical weights, or use `scripts/smoke_check.py`, which detects the checkpoint setting. Other exact hyperparameters for that historical checkpoint have not been verified.

## Packaging repairs

- Added explicit CLI parsing for previously ignored experiment parameters and corrected the epsilon schedule to use total environment steps.
- Added headless/no-video switches, a separate artifact directory and environment cleanup.
- Completed dependency declarations and removed an unused Gymnasium import. The active environment code uses its legacy Gym interface.

Only short model/environment/training checks were performed during packaging; full convergence and old benchmark results were not reproduced. The source still contains legacy wrapper conventions and shared-batch exploration; see [PROVENANCE.md](PROVENANCE.md) for ownership and evidence limits. Prioritised replay and Double DQN are not enabled. Initially created **private**; this is coursework built on supplied/reference code, not an original RL framework.
