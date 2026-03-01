import numpy as np
import random
from .transaction import Transaction

class BlockchainEnv:

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        n = self.cfg['num_peers']
        self.time = 0.0
        self.prev_mempool_mean = 0.0
        self.tx_counter = 0

        self.bandwidth = np.random.uniform(50, 200, n)
        self.mempool = [[] for _ in range(n)]

        coords = np.random.rand(n, 2)
        dist = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
        self.lat_matrix = 1 + dist * 4
        np.fill_diagonal(self.lat_matrix, 0)

        self.active_mask = np.ones(n)

        return self._state()

    def _state(self):
        avg_latency = np.mean(self.lat_matrix)
        latency_std = np.std(self.lat_matrix)

        mem_sizes = [len(m) for m in self.mempool]
        avg_mempool = np.mean(mem_sizes)

        mempool_growth = avg_mempool - self.prev_mempool_mean
        self.prev_mempool_mean = avg_mempool

        bw_mean = np.mean(self.bandwidth)
        num_peers = self.cfg['num_peers']

        return np.array([
            avg_latency,
            latency_std,
            avg_mempool,
            mempool_growth,
            bw_mean,
            num_peers
        ], dtype=np.float32)

    def _validator_churn(self):
        churn_prob = 0.02
        for i in range(len(self.active_mask)):
            if random.random() < churn_prob:
                self.active_mask[i] = 1 - self.active_mask[i]

        if np.sum(self.active_mask) == 0:
            self.active_mask[:] = 1

    def _generate_transactions(self):
        congestion_wave = 1 + 0.5 * np.sin(self.time / 50)
        burst_noise = np.random.uniform(0.8, 1.2)

        for peer in range(self.cfg['num_peers']):
            if self.active_mask[peer] == 0:
                continue

            arrivals = np.random.poisson(2 * congestion_wave * burst_noise)

            for _ in range(arrivals):
                tx = Transaction(
                    tx_id=self.tx_counter,
                    timestamp=self.time
                )
                self.tx_counter += 1
                self.mempool[peer].append(tx)

    def _latency_spike(self):
        if random.random() < 0.05:
            spike = np.random.uniform(1.5, 3.0)
            self.lat_matrix *= spike

    def step(self, action):

        n = self.cfg['num_peers']

        self._validator_churn()
        self._latency_spike()
        self._generate_transactions()

        mask = action[:n] > 0.5
        mask = mask * self.active_mask

        if np.sum(mask) == 0:
            mask[:] = self.active_mask

        block_size = 1 + action[-2] * 10
        interval = 0.5 + action[-1] * 5

        capacity = block_size
        processed = 0.0

        for peer in range(n):
            if self.active_mask[peer] == 0:
                continue

            new_pool = []
            for tx in self.mempool[peer]:
                if capacity >= tx.size_kb:
                    processed += tx.size_kb
                    capacity -= tx.size_kb
                else:
                    new_pool.append(tx)

            self.mempool[peer] = new_pool

        propagation = block_size / np.mean(self.bandwidth[mask.astype(bool)])
        network_factor = np.log(np.sum(mask) + 1)

        latency = propagation * network_factor

        fork_pressure = latency * (np.sum(mask) / n)
        orphan_rate = min(0.2, fork_pressure * 0.03)

        throughput = processed / interval
        efficiency = throughput / (latency + 1e-5)

        reward = (
            self.cfg['beta_efficiency'] * efficiency
            - self.cfg['alpha_latency'] * latency
            - self.cfg['lambda_orphan'] * orphan_rate
        )

        self.time += interval
        done = self.time >= self.cfg['max_time']

        info = {
            "latency": latency,
            "throughput": throughput,
            "orphan_rate": orphan_rate,
            "mempool": np.mean([len(m) for m in self.mempool]),
            "active_peers": np.sum(mask)
        }

        return self._state(), reward, done, info