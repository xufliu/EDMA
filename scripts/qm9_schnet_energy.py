import argparse
import itertools
import os
import pickle
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
for root in (PROJECT_ROOT, WORKSPACE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QM9 SchNet EDMA statistics"
    )
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--target-attr", type=int, default=0)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--k-list", default="2,3,4,5,6,7,8,9")
    parser.add_argument("--epochs", default="300,400,500")
    parser.add_argument("--node-feat-size", default="1,3,5,7")
    parser.add_argument("--node-feat-ent", default="0.0005,0.0015")
    parser.add_argument("--hinge-w", type=float, default=10.0)
    parser.add_argument("--temp", default="5.0,2.0")
    parser.add_argument(
        "--mask-logit-direction",
        default="one_minus_zero",
        choices=("one_minus_zero", "zero_minus_one"),
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--output",
        default="schnet_u_energy_final_hard_single2.pkl",
        help="Output pickle path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from torch_geometric.explain import Explainer
    from torch_geometric.loader import DataLoader

    from edma.evaluation.qm9_energy import (
        evaluate_qm9_energy_parameter_set,
        parse_float_list,
        parse_int_list,
    )
    from edma.explainers.schnet import SchNetEnergyInstanceExplainer
    from edma.models.pretrained import load_pretrained_schnet_qm9

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, splits = load_pretrained_schnet_qm9(
        data_path=args.data_path,
        target_attr=args.target_attr,
        device=device,
    )
    _, _, test_dataset = splits
    if args.test_size > 0:
        test_dataset = test_dataset[: args.test_size]

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    k_list = parse_int_list(args.k_list)
    epochs_list = parse_int_list(args.epochs)
    node_feat_size_list = parse_float_list(args.node_feat_size)
    node_feat_ent_list = parse_float_list(args.node_feat_ent)
    temp = parse_float_list(args.temp)
    if len(temp) != 2:
        raise ValueError("--temp must contain exactly two comma-separated values")

    parameter_combinations = list(
        itertools.product(epochs_list, node_feat_size_list, node_feat_ent_list)
    )

    best_mae = float("inf")
    best_parameters = None
    all_results = {}
    all_closeness = {}
    all_node_ranks = {}
    all_node_masks = {}

    for epoch, node_feat_size, node_feat_ent in parameter_combinations:
        parameter_key = f"{epoch},{node_feat_size},{node_feat_ent}"

        explainer = Explainer(
            model=model,
            algorithm=SchNetEnergyInstanceExplainer(
                epochs=epoch,
                lr=args.lr,
                node_feat_size=node_feat_size,
                node_feat_ent=node_feat_ent,
                hinge_w=args.hinge_w,
                temp=temp,
                mask_logit_direction=args.mask_logit_direction,
            ),
            explanation_type="model",
            node_mask_type="object",
            edge_mask_type=None,
            model_config=dict(
                mode="regression",
                task_level="graph",
                return_type="raw",
            ),
        )

        list_of_results, close_list, node_ranks, node_masks = (
            evaluate_qm9_energy_parameter_set(
            explainer=explainer,
            model=model,
            loader=test_loader,
            k_list=k_list,
            device=device,
        )
        )

        all_results[parameter_key] = list_of_results
        all_closeness[parameter_key] = close_list
        all_node_ranks[parameter_key] = node_ranks
        all_node_masks[parameter_key] = node_masks

        total_loss = sum(sum(maes) for maes in list_of_results.values())
        if total_loss < best_mae:
            best_mae = total_loss
            best_parameters = parameter_key
            print(best_parameters)
            print(best_mae)

    output = {
        "best_mae": best_mae,
        "best_parameters": best_parameters,
        "all_results": all_results,
        "all_closeness": all_closeness,
        "all_node_ranks": all_node_ranks,
        "all_node_masks": all_node_masks,
    }

    print("time:", time.time() - start_time)
    with open(args.output, "wb") as file_obj:
        pickle.dump(output, file_obj)

    print(best_mae)
    print(best_parameters)


if __name__ == "__main__":
    main()
