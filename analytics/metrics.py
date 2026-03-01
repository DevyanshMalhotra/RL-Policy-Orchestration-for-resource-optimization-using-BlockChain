import os
import pandas as pd

class Metrics:

    def __init__(self, output_dir):
        self.rows = []
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def record(self, info, reward=None):
        row = dict(info)
        if reward is not None:
            row["reward"] = reward
        self.rows.append(row)

    def save(self):
        df = pd.DataFrame(self.rows)
        path = os.path.join(self.output_dir, "metrics.csv")
        df.to_csv(path, index=False)
        return df