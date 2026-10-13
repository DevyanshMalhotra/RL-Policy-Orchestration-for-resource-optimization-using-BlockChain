# Reinforcement Learning for Blockchain Resource Optimization

A simulation-based reinforcement learning project exploring adaptive resource allocation in a simplified blockchain network. A PyTorch actor–critic agent adjusts **validator participation, block size and block interval** based on synthetic network observations. The project compares learned behavior with static and rule-based policies.

**Scope:** This is a research simulation—not a blockchain node, consensus implementation, live network controller, or deployed trading system.

## Architecture

```text
config.yaml
    |
    v
BlockchainEnv  ---> state vector (latency, mempool, bandwidth, peer count)
    |                                         |
    |                                         v
    |                                   RLAgent (PyTorch)
    |                                         |
    <---------- validator mask, block size, block interval
    |
    v
synthetic transaction processing + propagation estimates
    |
    v
throughput / confirmation latency / orphan-rate estimates
    |
    v
reward + metrics  ---> replay buffer ---> actor–critic updates
```

The agent is DDPG-style: an actor produces continuous actions, a critic estimates action value, and target networks are updated with a soft-update coefficient. The simulation includes random transaction arrivals, bandwidth variation and occasional network-latency spikes.

### Reward

The environment computes:

```text
efficiency = throughput / (latency + 1e-5)
reward = beta_efficiency * efficiency
         - alpha_latency * latency
         - lambda_orphan * orphan_rate
```

The configuration and implementation use **efficiency**, not throughput alone, as the positive reward term.

## Code map

| Location | Responsibility |
| --- | --- |
| `env/blockchain_env.py` | Simulated network state, actions and reward |
| `core/agent.py` | Actor, critic, target networks and learning |
| `core/replay_buffer.py` | Experience replay |
| `train.py` | Training and checkpoint saving |
| `evaluate.py` | RL vs. static vs. rule-based policies |
| `scaling.py` | Runs scaling analysis |
| `analytics/metrics.py` | Training metrics |
| `analytics/plots.py` | Evaluation charts |
| `config.yaml` | Experiment parameters |
| `output/` | Saved experimental results, images and checkpoints |

## Reproduce an experiment

From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python train.py
python evaluate.py
```

The checked-in `config.yaml` defaults to **5 peers**, 120 training episodes, and a 600-unit simulation horizon. Change `num_peers` and other parameters to explore different conditions. The evaluation script loads a checkpoint from `output/<num_peers>_nodes/` and writes result CSVs and comparison plots; train a matching configuration first if a checkpoint is missing.

Selected example artifacts are available in `output/5_nodes/`, `output/10_nodes/`, `output/15_nodes/`, `output/20_nodes/`, and `output/scaling/`. Charts are outputs of a stochastic simulation, not measurements of real blockchain networks.

## What this project demonstrates

- Translating network observations into an RL state vector.
- Mapping continuous policy outputs to validator participation and block parameters.
- Designing a combined performance / latency / orphan-rate reward.
- Training an off-policy actor–critic model with experience replay.
- Comparing policies across different simulated network sizes.

## Limitations

The environment uses simplified stochastic models of transaction arrivals, propagation, confirmation and orphaning. Some evaluation policy actions use fixed values while learned actions are normalized; comparisons should therefore be treated as **exploratory**, not as proof of production improvements. Results vary with random events and configuration. No production throughput, latency, security or scalability guarantees are made.
