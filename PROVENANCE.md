# Provenance and scope

Based on NUS CEG5301 Assignment 5 scaffolding. The supplied environment, wrapper, replay and vector-environment utilities reference Kchu/DeepRL_PyTorch and OpenAI Baselines/Gym; attribution comments are retained in the source. Student work completes the CNN/value head, epsilon-greedy action selection, DQN learning updates, SARSA/Q-learning updates and experiment analysis.

This is simulation coursework, not physical deployment or an original RL library. No licence is asserted over the whole collection. Initially uploaded as a private repository. Assignment handouts, student identifiers and duplicate copies under another course folder are not included.

Historical `data/plots/dqn_result_o_Pendulum.pkl` contains 69 logging points, ending at -198.41. `artifacts/monitor.csv` independently yields -198.41396088 for its last 100 completed episodes. The metric is a rolling training return, not a held-out evaluation. A separate report screenshot records a different run (118,000 steps, -191.7); that run's report is not included here.

The A/B/C comparison image is a historical artifact. Some CLI parameters were previously ignored, so the image cannot establish a verified acceleration or ablation claim. Packaging repairs make future parameterised runs explicit; they do not retrospectively validate old results. Prioritised replay and Double DQN are not enabled.
