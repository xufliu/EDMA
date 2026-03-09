import os
import os.path as osp
import time
import itertools
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from torch_geometric.explain import Explainer

from energy_explainer_instance_gpt import EnergyInstanceExplainer  # your algorithm

# ---------------------------
# Config
# ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "QM9")
dataset = QM9(path)

target_attr = 0  # dipole_moment
schnet, splits = SchNet.from_qm9_pretrained(path, dataset, target_attr)
schnet = schnet.to(device)
schnet.eval()

train_dataset, val_dataset, test_dataset = splits
test_dataset = test_dataset[:1024]

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Evaluate MAE@k for k in [2..9]
K_LIST = list(range(2, 10))

# Hyperparam grid (example; adjust)
epochs_list = [200, 400, 800]
node_feat_size_list = [0.001, 0.003, 0.01, 0.03, 0.1]
hinge_w_list = [0, 1, 3, 10]
temp_start_list = [2.0, 5.0]
temp_end_list = [0.5, 0.2, 0.1]

param_grid = list(
    itertools.product(
        epochs_list, node_feat_size_list, hinge_w_list, temp_start_list, temp_end_list
    )
)


# ---------------------------
# Helper: compute MAE@k
# ---------------------------
@torch.no_grad()
def mae_at_k_for_graph(schnet_model, z_g, pos_g, y_true_g, node_mask_g, k: int):
    # node_mask_g: [N_g] float
    # Select top-k nodes:
    _, ranked = torch.sort(node_mask_g, descending=True)
    sel = ranked[:k].sort().values

    # Subgraph forward (single graph => batch=None):
    y_sub = schnet_model(z_g[sel], pos_g[sel], batch=None)  # [1,1] or [1]
    y_sub = y_sub.view(-1)

    y_true_g = y_true_g.view(-1)

    return torch.abs(y_sub - y_true_g).item(), sel.numel()


def run_eval(explainer, loader, schnet_model, target_attr: int, k_list):
    results = {k: [] for k in k_list}
    mask_stats = {
        "mean_m": [],
        "bin_frac": [],  # fraction of nodes with m<0.05 or m>0.95
        "num_nodes": [],
    }

    t0 = time.time()

    for data in loader:
        data = data.to(device)

        # Ground-truth labels for this batch:
        # QM9 stores y as [num_graphs, 12]
        y_true = data.y[:, target_attr].view(-1, 1)  # [B,1]

        # Run explainer (your algorithm optimizes masks using label fidelity):
        explanation = explainer(data.z, data.pos, target=y_true, batch=data.batch)

        # Your algorithm returns node_mask as [num_nodes] or [num_nodes,1]
        node_mask_all = explanation.node_mask
        if node_mask_all.dim() == 2:
            node_mask_all = node_mask_all.view(-1)

        # Debug stats across the whole batch:
        with torch.no_grad():
            mean_m = float(node_mask_all.mean().item())
            bin_frac = float(
                ((node_mask_all < 0.05) | (node_mask_all > 0.95)).float().mean().item()
            )
            mask_stats["mean_m"].append(mean_m)
            mask_stats["bin_frac"].append(bin_frac)
            mask_stats["num_nodes"].append(int(node_mask_all.numel()))

        # Per-graph evaluation:
        num_graphs = int(data.num_graphs)
        for g in range(num_graphs):
            node_idx = (data.batch == g).nonzero(as_tuple=False).view(-1)

            z_g = data.z[node_idx]
            pos_g = data.pos[node_idx]
            y_true_g = y_true[g]  # [1,1]
            node_mask_g = node_mask_all[node_idx]  # [N_g]

            for k in k_list:
                if node_idx.numel() < k:
                    continue
                mae_k, _ = mae_at_k_for_graph(
                    schnet_model, z_g, pos_g, y_true_g, node_mask_g, k
                )
                results[k].append(mae_k)

    elapsed = time.time() - t0
    return results, mask_stats, elapsed


# ---------------------------
# Main grid search
# ---------------------------
best_score = float("inf")
best_params = None
all_runs = {}

for epochs, node_feat_size, hinge_w, t0, t1 in param_grid:
    # print(
    #     f"\n=== Run: epochs={epochs}, node_feat_size={node_feat_size}, hinge_w={hinge_w}, temp=({t0}->{t1}) ==="
    # )

    explainer = Explainer(
        model=schnet,
        algorithm=EnergyInstanceExplainer(
            epochs=epochs,
            node_feat_size=node_feat_size,
            hinge_w=hinge_w,
            temp=[t0, t1],
            log_every=50,
        ),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type=None,
        model_config=dict(mode="regression", task_level="graph", return_type="raw"),
    )

    results, mask_stats, elapsed = run_eval(
        explainer, test_loader, schnet, target_attr, K_LIST
    )

    # Aggregate score: sum over k of mean MAE@k (you can change objective)
    mean_maes = {
        k: float(np.mean(v)) if len(v) > 0 else float("nan") for k, v in results.items()
    }
    score = sum(mean_maes[k] for k in K_LIST if not np.isnan(mean_maes[k]))

    print("Mean MAE@k:", mean_maes)
    print(f"Score={score:.6f}  time={elapsed:.1f}s")
    print(
        f"Mask stats: mean(m)={np.mean(mask_stats['mean_m']):.4f}, bin_frac={np.mean(mask_stats['bin_frac']):.4f}"
    )

    run_key = f"epochs={epochs}|size={node_feat_size}|hinge={hinge_w}|temp={t0}->{t1}"
    all_runs[run_key] = {
        "mean_mae_at_k": mean_maes,
        "raw_mae_at_k": results,
        "mask_stats": mask_stats,
        "elapsed_sec": elapsed,
    }

    if score < best_score:
        best_score = score
        best_params = run_key
        print(">>> New best:", best_params, "score=", best_score)

# Save
out = {
    "best_score": best_score,
    "best_params": best_params,
    "all_runs": all_runs,
}
with open("qm9_schnet_edma_label_fidelity.pkl", "wb") as f:
    pickle.dump(out, f)

print("\nDONE")
print("Best:", best_params, "score=", best_score)
