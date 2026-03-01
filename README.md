## System Overview

This system simulates a **decentralized blockchain network** controlled by a reinforcement learning (RL) agent. Each node in the network operates independently while contributing to a shared global state used for learning and decision-making.

Each node performs the following tasks:

- Collects local metrics  
- Reports **latency**, **mempool depth**, and **bandwidth**  
- Contributes to a **global state vector** consumed by the RL agent  

---

## Reinforcement Learning Policy

The RL agent observes the global state and determines key protocol-level parameters:

- Active validator set  
- Block size  
- Block interval  

The policy is learned using an **Actor–Critic (DDPG-style)** algorithm.

---

## Environment Execution

Given the agent’s actions, the environment executes the blockchain protocol by:

- Sampling peers  
- Assembling blocks  
- Propagating blocks across the network  

During execution, the environment measures:

- Confirmation latency  
- Throughput  
- Orphan rate  

---

## Reward Function

The reward signal is computed at each step as:

```

r = β · throughput − α · latency − λ · orphan_rate

```

Where:

- `β` controls the importance of throughput  
- `α` penalizes confirmation latency  
- `λ` penalizes orphaned blocks  

Reward calculation is performed **inside the environment step function**.

---

## Learning Process

- The agent updates its policy using an Actor–Critic (DDPG-style) method  
- Transaction size is fixed at **256 bytes** to ensure consistent propagation behavior across experiments  

---

## Output Organization

All experiment outputs are organized using the following directory structure:

```

output/<num_peers>_nodes/

```

Network scaling experiment results are stored separately under:

```

output/scaling/

```

---

## Function Mapping to Architecture

The following table maps core functions to their responsibilities within the system architecture:

| Function | Responsibility |
|--------|----------------|
| `BlockchainEnv._state()` | Collect local metrics |
| `RLAgent.act()` | RL policy decision |
| `BlockchainEnv.step()` | Execute actions and measure outcomes |
| Reward calculation | Implemented inside `step()` |
| `RLAgent.learn()` | Policy update |
| `run_training()` | Learning loop |
| `run_evaluation()` | Policy comparison |
| `plot_research()` | Outcome visualization |
| `plot_scaling()` | Network scaling analysis |



