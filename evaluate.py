import yaml
import numpy as np
import pandas as pd
import os

from env.blockchain_env import BlockchainEnv
from core.agent import RLAgent
from analytics.plots import plot_research

def static_policy(state, cfg):
    n = cfg['num_peers']
    mask = np.ones(n)
    block_size = 5.0
    interval = 3.0
    return np.concatenate([mask, [block_size, interval]])

def rule_policy(state, cfg):
    n = cfg['num_peers']
    avg_lat, lat_std, mem, mem_growth, bw, peers = state
    mask = np.zeros(n)
    mask[: n // 2] = 1
    block_size = min(6.0, mem + 1)
    interval = 2.0 if mem > 10 else 4.0
    return np.concatenate([mask, [block_size, interval]])

def run_policy(env, agent, cfg, policy_name):
    records = []

    for _ in range(5):
        s = env.reset()
        done = False

        while not done:

            if policy_name == "rl":
                a = agent.act(s)
            elif policy_name == "static":
                a = static_policy(s, cfg)
            else:
                a = rule_policy(s, cfg)

            s2, r, done, info = env.step(a)
            records.append(info)
            s = s2

    df = pd.DataFrame(records)
    df.to_csv(f"results_{policy_name}_{cfg['num_peers']}.csv", index=False)
    return df

def run_evaluation(cfg):

    output_dir = os.path.join("output", f"{cfg['num_peers']}_nodes")
    os.makedirs(output_dir, exist_ok=True)

    env = BlockchainEnv(cfg)
    state_dim = len(env.reset())
    agent = RLAgent(state_dim, cfg['num_peers'] + 2, cfg)
    agent.load(os.path.join(output_dir, f"rl_agent_peers_{cfg['num_peers']}.pt"))

    df_rl = run_policy(env, agent, cfg, "rl")
    df_static = run_policy(env, agent, cfg, "static")
    df_rule = run_policy(env, agent, cfg, "rule")

    df_rl.to_csv(os.path.join(output_dir, f"results_rl_{cfg['num_peers']}.csv"), index=False)
    df_static.to_csv(os.path.join(output_dir, f"results_static_{cfg['num_peers']}.csv"), index=False)
    df_rule.to_csv(os.path.join(output_dir, f"results_rule_{cfg['num_peers']}.csv"), index=False)

    plot_research(df_rl, df_static, df_rule, output_dir)

    return df_rl

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    run_evaluation(cfg)