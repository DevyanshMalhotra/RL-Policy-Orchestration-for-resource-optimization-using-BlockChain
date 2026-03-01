import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_research(df_rl, df_static, df_rule, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.plot(df_rl['throughput'].rolling(50).mean(), label="RL-Orch")
    plt.plot(df_static['throughput'].rolling(50).mean(), label="Static")
    plt.plot(df_rule['throughput'].rolling(50).mean(), label="RuleBased")
    plt.legend()
    plt.title("Throughput Optimization Over Time")
    plt.savefig(os.path.join(output_dir, "g1_throughput_compare.png"))

    plt.figure()
    plt.plot(df_rl['latency'].rolling(50).mean(), label="RL-Orch")
    plt.plot(df_static['latency'].rolling(50).mean(), label="Static")
    plt.plot(df_rule['latency'].rolling(50).mean(), label="RuleBased")
    plt.legend()
    plt.title("Latency Reduction Behaviour")
    plt.savefig(os.path.join(output_dir, "g2_latency_compare.png"))

    plt.figure()
    plt.plot(df_rl['orphan_rate'], alpha=0.6, label="RL-Orch")
    plt.plot(df_static['orphan_rate'], alpha=0.6, label="Static")
    plt.legend()
    plt.title("Fork/Orphan Stability")
    plt.savefig(os.path.join(output_dir, "g3_orphan_compare.png"))

    plt.figure()
    plt.scatter(df_rl['latency'], df_rl['throughput'], alpha=0.4, label="RL")
    plt.scatter(df_static['latency'], df_static['throughput'], alpha=0.4, label="Static")
    plt.legend()
    plt.title("Latency vs Throughput Tradeoff")
    plt.savefig(os.path.join(output_dir, "g4_tradeoff.png"))

    plt.figure()
    df_rl['active_peers'].plot(title="Adaptive Validator Selection")
    plt.savefig(os.path.join(output_dir, "g5_peers.png"))

    plt.figure()
    df_rl['mempool'].rolling(50).mean().plot(title="Mempool Stabilisation")
    plt.savefig(os.path.join(output_dir, "g6_mempool.png"))

    plt.figure()
    metrics = ["latency", "throughput", "orphan_rate"]
    rl_vals = df_rl[metrics].mean()
    static_vals = df_static[metrics].mean()
    rule_vals = df_rule[metrics].mean()
    x = range(len(metrics))
    plt.bar([i - 0.2 for i in x], rl_vals, width=0.2, label="RL")
    plt.bar(x, static_vals, width=0.2, label="Static")
    plt.bar([i + 0.2 for i in x], rule_vals, width=0.2, label="Rule")
    plt.xticks(x, metrics)
    plt.legend()
    plt.title("Final Policy Comparison")
    plt.savefig(os.path.join(output_dir, "g7_final_bar.png"))

    try:
        df_metrics = pd.read_csv(os.path.join(output_dir, "metrics.csv"))
        if "reward" in df_metrics.columns:
            plt.figure()
            df_metrics['reward'].rolling(50).mean().plot()
            plt.title("Policy Convergence Behaviour")
            plt.savefig(os.path.join(output_dir, "g8_convergence.png"))
    except:
        pass

def plot_scaling(df, output_dir):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.plot(df['peers'], df['throughput'], marker='o')
    plt.title("Throughput vs Network Size")
    plt.xlabel("Number of Peers")
    plt.ylabel("Throughput")
    plt.savefig(os.path.join(output_dir, "g9_scaling_throughput.png"))

    plt.figure()
    plt.plot(df['peers'], df['latency'], marker='o')
    plt.title("Latency vs Network Size")
    plt.xlabel("Number of Peers")
    plt.ylabel("Latency")
    plt.savefig(os.path.join(output_dir, "g10_scaling_latency.png"))
    