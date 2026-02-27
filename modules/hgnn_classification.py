"""
HGNN Node Classification Module — Adapted from Hypergraph-HGNN
================================================================
Based on: "Hypergraph Neural Networks" (Feng et al., AAAI 2019)
Embedding: TF-IDF (simple, no extra deps)
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import OrderedDict

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ──────────────────────── Data Loading ────────────────────────

def load_node_attributes(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_node_attributes_from_text(text):
    return json.loads(text)

def load_hyperedges(path):
    hyperedges = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            nodes = list(set(int(x) for x in line.split()))
            if len(nodes) >= 2:
                hyperedges.append(sorted(nodes))
    return hyperedges

def load_hyperedges_from_lines(lines):
    hyperedges = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            nodes = list(set(int(x) for x in line.split()))
            if len(nodes) >= 2:
                hyperedges.append(sorted(nodes))
        except ValueError:
            continue
    return hyperedges

def get_all_nodes(hyperedges):
    nodes = set()
    for he in hyperedges:
        nodes.update(he)
    return sorted(nodes)


# ──────────────────────── Feature Encoding ────────────────────────

def build_text_for_node(attr):
    parts = [
        attr.get('gender', ''),
        attr.get('education', ''),
        attr.get('workplace', ''),
    ]
    venues = attr.get('representative_venues', [])
    if venues:
        parts.append(' '.join(venues))
    skills = attr.get('top_skills', [])
    if skills:
        parts.append(' '.join(skills))
    excluded = {
        'gender', 'education', 'age', 'workplace', 'major',
        'research_interest', 'representative_venues',
        'professional_membership', 'top_skills', 'h_index',
        'personal_description'
    }
    for key, val in attr.items():
        if key not in excluded:
            if isinstance(val, str):
                parts.append(val)
            elif isinstance(val, list):
                parts.append(' '.join(str(v) for v in val))
    return ' '.join(p for p in parts if p)


def encode_tfidf(node_attrs, all_node_ids, max_features=128):
    texts, ages, h_indices, majors = [], [], [], []
    for nid in all_node_ids:
        attr = node_attrs.get(str(nid), {})
        texts.append(build_text_for_node(attr))
        ages.append(float(attr.get('age', 30)))
        h_indices.append(float(attr.get('h_index', 0)))
        majors.append(attr.get('major', 'Unknown'))
    vectorizer = TfidfVectorizer(max_features=max_features,
                                 stop_words='english', sublinear_tf=True)
    text_feat = vectorizer.fit_transform(texts).toarray()
    scaler = StandardScaler()
    num_feat = scaler.fit_transform(np.column_stack([ages, h_indices]))
    features = np.hstack([text_feat, num_feat]).astype(np.float32)
    le = LabelEncoder()
    labels = le.fit_transform(majors)
    return features, labels, le


# ──────────────────────── Hypergraph Construction ────────────────────────

def build_incidence_matrix(hyperedges, num_nodes, node_to_idx):
    H = np.zeros((num_nodes, len(hyperedges)), dtype=np.float32)
    for j, he in enumerate(hyperedges):
        for nid in he:
            if nid in node_to_idx:
                H[node_to_idx[nid], j] = 1.0
    return H

def compute_G(H):
    Dv = H.sum(axis=1)
    De = H.sum(axis=0)
    Dv_inv_sqrt = np.zeros_like(Dv)
    Dv_inv_sqrt[Dv > 0] = 1.0 / np.sqrt(Dv[Dv > 0])
    De_inv = np.zeros_like(De)
    De_inv[De > 0] = 1.0 / De[De > 0]
    G = Dv_inv_sqrt[:, None] * (H * De_inv[None, :]) @ H.T * Dv_inv_sqrt[None, :]
    return G.astype(np.float32)


# ──────────────────────── HGNN Model ────────────────────────

class HGNNConv(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ch, out_ch))
        self.bias = nn.Parameter(torch.Tensor(out_ch)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, X, G):
        X = X @ self.weight
        X = G @ X
        if self.bias is not None:
            X = X + self.bias
        return X


class HGNN(nn.Module):
    def __init__(self, in_ch, hid, n_class, dropout=0.5):
        super().__init__()
        self.conv1 = HGNNConv(in_ch, hid)
        self.conv2 = HGNNConv(hid, n_class)
        self.dropout = dropout

    def forward(self, X, G):
        X = F.relu(self.conv1(X, G))
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = self.conv2(X, G)
        return X


# ──────────────────────── Training ────────────────────────

def train_and_evaluate(features, G, labels, train_mask, test_mask,
                       num_classes, device, epochs=200, lr=0.01,
                       hidden_dim=128, dropout=0.5, weight_decay=5e-4,
                       run_id=0):
    torch.manual_seed(SEED + run_id)
    np.random.seed(SEED + run_id)

    X = torch.FloatTensor(features).to(device)
    G_t = torch.FloatTensor(G).to(device)
    y = torch.LongTensor(labels).to(device)
    tr = torch.BoolTensor(train_mask).to(device)
    te = torch.BoolTensor(test_mask).to(device)

    model = HGNN(features.shape[1], hidden_dim, num_classes, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    train_losses = []
    test_accs = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X, G_t)
        loss = criterion(out[tr], y[tr])
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                pred = model(X, G_t).argmax(dim=1)
                acc = (pred[te] == y[te]).float().mean().item()
                test_accs.append({"epoch": epoch, "accuracy": round(acc, 4)})
                if acc > best_acc:
                    best_acc = acc

    return best_acc, train_losses, test_accs


# ──────────────────────── Experiment Runner ────────────────────────

def run_hgnn_experiment(hyperedge_files, attr_data, all_node_ids=None,
                        num_train=300, num_test=100, epochs=200,
                        hidden_dim=128, num_runs=5, lr=0.01):
    """
    Run HGNN node classification experiment.

    Args:
        hyperedge_files: dict {name: list_of_hyperedges}
        attr_data: dict of node attributes (from JSON)
        all_node_ids: list of node IDs (if None, derived from attributes)
        num_train, num_test: train/test split sizes
        epochs, hidden_dim, num_runs, lr: training params

    Returns:
        JSON-serializable results dict
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if all_node_ids is None:
        all_node_ids = sorted(int(k) for k in attr_data.keys())

    # Encode features
    features, labels, le = encode_tfidf(attr_data, all_node_ids)
    num_classes = len(le.classes_)
    node_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}
    num_nodes = len(all_node_ids)

    # Fixed split: find nodes that appear in any hyperedge
    all_he_nodes = set()
    for name, hes in hyperedge_files.items():
        for he in hes:
            all_he_nodes.update(he)
    valid_nodes = sorted(set(all_node_ids) & all_he_nodes)

    if len(valid_nodes) < num_train + num_test:
        num_test = max(10, len(valid_nodes) // 4)
        num_train = len(valid_nodes) - num_test

    shuffled = valid_nodes.copy()
    random.Random(SEED).shuffle(shuffled)
    test_ids = sorted(shuffled[:num_test])
    train_ids = sorted(shuffled[num_test:num_test + num_train])

    results = OrderedDict()

    for exp_name, hyperedges in hyperedge_files.items():
        H = build_incidence_matrix(hyperedges, num_nodes, node_to_idx)
        G = compute_G(H)
        G = G + 0.05 * np.eye(num_nodes, dtype=np.float32)

        train_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        for nid in train_ids:
            if nid in node_to_idx:
                train_mask[node_to_idx[nid]] = True
        for nid in test_ids:
            if nid in node_to_idx:
                test_mask[node_to_idx[nid]] = True

        accs = []
        all_losses = []
        all_test_accs = []
        for r in range(num_runs):
            acc, losses, taccs = train_and_evaluate(
                features, G, labels, train_mask, test_mask,
                num_classes, device, epochs=epochs, lr=lr,
                hidden_dim=hidden_dim, run_id=r
            )
            accs.append(acc)
            if r == 0:
                all_losses = losses
                all_test_accs = taccs

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        results[exp_name] = {
            "mean_accuracy": round(mean_acc * 100, 2),
            "std_accuracy": round(std_acc * 100, 2),
            "num_hyperedges": len(hyperedges),
            "train_losses": all_losses[::5],  # subsample for plotting
            "test_accs": all_test_accs,
        }

    return {
        "experiments": results,
        "config": {
            "num_train": num_train,
            "num_test": num_test,
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "num_runs": num_runs,
            "num_classes": num_classes,
            "feature_dim": features.shape[1],
            "device": str(device),
        }
    }
