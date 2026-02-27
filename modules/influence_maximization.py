"""
Influence Maximization Module — Adapted from Hypergraph_IM
==========================================================
Diffusion models: HIC (Hypergraph IC), HLT (Hypergraph LT), WC (Weighted Cascade)
Seed strategies:  Random, Degree, PageRank
"""

import random
import time
import numpy as np
from collections import defaultdict
from typing import List, Dict, Set

# ────────────────────────── Hypergraph ──────────────────────────

class Hypergraph:
    def __init__(self):
        self.hyperedges: List[List[int]] = []
        self.nodes: Set[int] = set()
        self.node_to_edges: Dict[int, List[int]] = defaultdict(list)
        self._edge_sizes: List[int] = []
        self._node_degree: Dict[int, int] = {}
        self._mean_degree: float = 0.0

    @classmethod
    def from_file(cls, filepath):
        hg = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    nodes = list(map(int, line.split()))
                except ValueError:
                    continue
                if nodes:
                    eidx = len(hg.hyperedges)
                    hg.hyperedges.append(nodes)
                    hg._edge_sizes.append(len(nodes))
                    for n in nodes:
                        hg.nodes.add(n)
                        hg.node_to_edges[n].append(eidx)
        for n in hg.nodes:
            hg._node_degree[n] = len(hg.node_to_edges[n])
        total = sum(hg._edge_sizes)
        hg._mean_degree = total / max(len(hg.nodes), 1)
        return hg

    @classmethod
    def from_lines(cls, lines):
        hg = cls()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                nodes = list(map(int, line.split()))
            except ValueError:
                continue
            if nodes:
                eidx = len(hg.hyperedges)
                hg.hyperedges.append(nodes)
                hg._edge_sizes.append(len(nodes))
                for n in nodes:
                    hg.nodes.add(n)
                    hg.node_to_edges[n].append(eidx)
        for n in hg.nodes:
            hg._node_degree[n] = len(hg.node_to_edges[n])
        total = sum(hg._edge_sizes)
        hg._mean_degree = total / max(len(hg.nodes), 1)
        return hg

    @property
    def num_nodes(self): return len(self.nodes)
    @property
    def num_edges(self): return len(self.hyperedges)

    def auto_ic_prob(self):
        avg_es = np.mean(self._edge_sizes) if self._edge_sizes else 2.0
        p = min(0.95, 0.4 + 0.06 * (avg_es - 2))
        return max(0.3, p)

    def summary(self):
        s = self._edge_sizes
        if not s:
            return {"nodes": 0, "edges": 0}
        return {
            "nodes": self.num_nodes, "edges": self.num_edges,
            "avg_edge_size": round(float(np.mean(s)), 2),
            "mean_degree": round(self._mean_degree, 2),
        }

# ────────────────────────── Diffusion Models ──────────────────────────

def ic_diffusion(hg, seeds, p, mc_rounds=50):
    """Hypergraph Independent Cascade (AND-gate)."""
    total = 0
    ne = hg.node_to_edges
    edges = hg.hyperedges
    esizes = hg._edge_sizes

    for _ in range(mc_rounds):
        active = set(seeds)
        edge_ac = [0] * len(edges)
        for s in seeds:
            for eidx in ne.get(s, []):
                edge_ac[eidx] += 1
        ready = set()
        for s in seeds:
            for eidx in ne.get(s, []):
                if esizes[eidx] > 1 and edge_ac[eidx] == esizes[eidx] - 1:
                    ready.add(eidx)
        attempted = set()
        while ready:
            newly = []
            for eidx in ready:
                if esizes[eidx] <= 1 or edge_ac[eidx] < esizes[eidx] - 1:
                    continue
                for v in edges[eidx]:
                    if v not in active:
                        key = (eidx, v)
                        if key not in attempted:
                            attempted.add(key)
                            if random.random() < p:
                                newly.append(v)
                        break
            if not newly:
                break
            ready = set()
            for v in newly:
                if v in active:
                    continue
                active.add(v)
                for eidx in ne.get(v, []):
                    edge_ac[eidx] += 1
                    if esizes[eidx] > 1 and edge_ac[eidx] == esizes[eidx] - 1:
                        ready.add(eidx)
        total += len(active)
    return total / mc_rounds


def wc_diffusion(hg, seeds, mc_rounds=50):
    """Weighted Cascade on hypergraph."""
    total = 0
    ne = hg.node_to_edges
    edges = hg.hyperedges
    esizes = hg._edge_sizes
    nd = hg._node_degree
    for _ in range(mc_rounds):
        active = set(seeds)
        frontier = list(seeds)
        while frontier:
            nf = []
            for u in frontier:
                for eidx in ne.get(u, []):
                    es = esizes[eidx]
                    if es <= 1:
                        continue
                    for v in edges[eidx]:
                        if v not in active:
                            prob = 1.0 / (max(nd.get(v, 1), 1) * (es - 1))
                            if random.random() < prob:
                                active.add(v)
                                nf.append(v)
            frontier = nf
        total += len(active)
    return total / mc_rounds


def lt_diffusion(hg, seeds, mc_rounds=50, theta_max=0.14):
    """Hypergraph Linear Threshold."""
    total = 0
    ne = hg.node_to_edges
    edges = hg.hyperedges
    esizes = hg._edge_sizes
    nd = hg._node_degree
    num_edges = len(edges)

    for _ in range(mc_rounds):
        thresholds = {n: random.uniform(0, theta_max) for n in hg.nodes}
        active = set(seeds)
        edge_ac = [0] * num_edges
        for s in seeds:
            for eidx in ne.get(s, []):
                edge_ac[eidx] += 1
        candidates = set()
        for s in seeds:
            for eidx in ne.get(s, []):
                for v in edges[eidx]:
                    if v not in active:
                        candidates.add(v)
        while candidates:
            new_active = []
            for v in candidates:
                if v in active:
                    continue
                dv = nd.get(v, 0)
                if dv == 0:
                    continue
                influence = sum(edge_ac[eidx] / esizes[eidx]
                                for eidx in ne.get(v, []) if esizes[eidx] > 1)
                influence /= dv
                if influence >= thresholds[v]:
                    new_active.append(v)
            if not new_active:
                break
            nc = set()
            for v in new_active:
                active.add(v)
                for eidx in ne.get(v, []):
                    edge_ac[eidx] += 1
                    for u in edges[eidx]:
                        if u not in active:
                            nc.add(u)
            candidates = nc
        total += len(active)
    return total / mc_rounds


MODELS = {
    'IC': {'func': ic_diffusion, 'name': 'Hypergraph IC (HIC)'},
    'WC': {'func': wc_diffusion, 'name': 'Weighted Cascade (WC)'},
    'LT': {'func': lt_diffusion, 'name': 'Hypergraph LT (HLT)'},
}

# ────────────────────────── Seed Strategies ──────────────────────────

def select_random(hg, k):
    return random.sample(list(hg.nodes), min(k, hg.num_nodes))

def select_degree(hg, k):
    ranked = sorted(hg.nodes, key=lambda n: (-hg._node_degree.get(n, 0), n))
    return ranked[:k]

def select_pagerank(hg, k, alpha=0.85, max_iter=80, tol=1e-6):
    nodes = sorted(hg.nodes)
    n = len(nodes)
    if n == 0:
        return []
    idx = {nd: i for i, nd in enumerate(nodes)}
    out_w = np.zeros(n)
    adj = defaultdict(float)
    for edge in hg.hyperedges:
        if len(edge) <= 1:
            continue
        w = 1.0 / (len(edge) - 1)
        for u in edge:
            ui = idx[u]
            for v in edge:
                if u != v:
                    vi = idx[v]
                    adj[(ui, vi)] += w
                    out_w[ui] += w
    pr = np.ones(n) / n
    for _ in range(max_iter):
        new_pr = np.full(n, (1 - alpha) / n)
        for (i, j), w in adj.items():
            if out_w[i] > 0:
                new_pr[j] += alpha * pr[i] * w / out_w[i]
        if np.sum(np.abs(new_pr - pr)) < tol:
            break
        pr = new_pr
    ranked = sorted(range(n), key=lambda i: -pr[i])
    return [nodes[i] for i in ranked[:k]]


STRATEGIES = {
    'Random':   {'color': '#2196F3', 'marker': 'o', 'func': select_random},
    'Degree':   {'color': '#4CAF50', 'marker': 's', 'func': select_degree},
    'PageRank': {'color': '#FF9800', 'marker': '^', 'func': select_pagerank},
}

# ────────────────────────── Experiment Runner ──────────────────────────

def run_im_experiment(hg, k_values=None, mc=30, models=None, strategies=None,
                      ic_prob=0, lt_theta=0.14, rand_repeat=3):
    """
    Run IM experiment on a single hypergraph.
    Returns JSON-serializable results.
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]
    k_values = [k for k in k_values if k <= hg.num_nodes]
    if not k_values:
        k_values = [min(hg.num_nodes, 5)]

    if models is None:
        models = ['LT', 'IC', 'WC']
    if strategies is None:
        strategies = ['Random', 'Degree', 'PageRank']

    max_k = max(k_values)
    max_k = min(max_k, hg.num_nodes)

    # Pre-compute seeds
    seeds_cache = {}
    for sn in strategies:
        if sn == 'Random':
            continue
        sfunc = STRATEGIES[sn]['func']
        seeds_cache[sn] = sfunc(hg, max_k)

    auto_p = hg.auto_ic_prob()
    actual_ic_prob = ic_prob if ic_prob > 0 else auto_p

    model_params = {
        'IC': {'p': actual_ic_prob, 'mc_rounds': mc},
        'WC': {'mc_rounds': mc},
        'LT': {'mc_rounds': mc, 'theta_max': lt_theta},
    }

    results = {}
    for mk in models:
        if mk not in MODELS:
            continue
        func = MODELS[mk]['func']
        params = model_params.get(mk, {})
        results[mk] = {}

        for sn in strategies:
            results[mk][sn] = {}
            for k in k_values:
                if sn == 'Random':
                    vals = []
                    for _ in range(rand_repeat):
                        s = set(select_random(hg, k))
                        vals.append(func(hg, s, **params))
                    sp = float(np.mean(vals))
                else:
                    s = set(seeds_cache[sn][:k])
                    sp = func(hg, s, **params)
                results[mk][sn][k] = round(sp, 2)

    return {
        "results": results,
        "k_values": k_values,
        "models": models,
        "strategies": strategies,
        "ic_prob_used": actual_ic_prob,
        "lt_theta": lt_theta,
        "summary": hg.summary(),
    }
