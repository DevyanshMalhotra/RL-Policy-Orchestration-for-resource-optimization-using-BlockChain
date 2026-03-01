import yaml
import pandas as pd
import os

from train import run_training
from evaluate import run_evaluation
from analytics.plots import plot_scaling

peer_list = [5,10,15,20]

results=[]

for peers in peer_list:

    print(f"\nRunning Experiment for {peers} Peers:")

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg['num_peers'] = peers

    run_training(cfg)

    df_rl = run_evaluation(cfg)

    results.append({
        "peers": peers,
        "throughput": df_rl['throughput'].mean(),
        "latency": df_rl['latency'].mean(),
        "orphan_rate": df_rl['orphan_rate'].mean()
    })


df_scale = pd.DataFrame(results)

scaling_dir = os.path.join("output", "scaling")
os.makedirs(scaling_dir, exist_ok=True)

df_scale.to_csv(os.path.join(scaling_dir, "scaling_results.csv"), index=False)

plot_scaling(df_scale, scaling_dir)

print("\nScaling experiments finished.")