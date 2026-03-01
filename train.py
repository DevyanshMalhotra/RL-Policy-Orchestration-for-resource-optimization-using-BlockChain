import yaml
import numpy as np
import random
import torch
import os

from env.blockchain_env import BlockchainEnv
from core.agent import RLAgent
from core.replay_buffer import ReplayBuffer
from analytics.metrics import Metrics

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

def run_training(cfg):

    output_dir = os.path.join("output", f"{cfg['num_peers']}_nodes")
    os.makedirs(output_dir, exist_ok=True)

    env = BlockchainEnv(cfg)
    state_dim = len(env.reset())
    agent = RLAgent(state_dim, cfg['num_peers'] + 2, cfg)
    buffer = ReplayBuffer(cfg['buffer_size'])
    metrics = Metrics(output_dir)

    for ep in range(cfg['episodes']):
        s = env.reset()
        done = False
        episode_reward = 0

        while not done:
            a = agent.act(s)
            s2, r, done, info = env.step(a)

            buffer.add(s, a, r, s2, done)

            if len(buffer.data) > cfg['batch_size']:
                batch = buffer.sample(cfg['batch_size'])
                agent.learn(batch)

            metrics.record(info, r)
            episode_reward += r
            s = s2

        print(f"Peers {cfg['num_peers']} | Episode {ep+1} Reward: {episode_reward:.3f}")

    metrics.save()
    agent.save(os.path.join(output_dir, f"rl_agent_peers_{cfg['num_peers']}.pt"))

if __name__ == "__main__":
    run_training(cfg)