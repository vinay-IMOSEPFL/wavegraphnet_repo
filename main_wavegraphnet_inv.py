"""
WaveGraphNet — Inverse Only training.

Pure inverse pre-training for N epochs. No forward model involved.
Checkpoint saved when val MAE improves. Result logged to results_mae.json.

Usage:
  python main_wavegraphnet_inv.py --split B2 --seed 0 \
    --inv_hidden_dim 256 --num_gnn_proc_layers 4 --gat_heads 16 \
    --epochs 900 --lr 1e-4 --batch_size 8
"""
import argparse, sys, random, pickle
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from utils.splits import get_train_val_test_ids
from utils.data_loader import CoupledModelDataset
from utils.precompute import build_all_stats, N_ATTENTION_FREQS
from utils.checkpointer import save_checkpoint, checkpoint_path
from models.wavegraphnet import GNN_inv_HierarchicalAttention, DirectPathAttenuationGNN
from utils.logger import log_result, log_mae_result, EpochLogger, evaluate_test

_TQDM_DISABLE = not sys.stdout.isatty()
PLATE_MM      = 500.0
LABEL         = "WaveGraphNet (Inverse Only)"


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def euclidean_mae(model, loader, device):
    model.eval(); total = 0.0; count = 0
    with torch.no_grad():
        for batch in loader:
            di   = batch["data_inv"].to(device)
            yt   = batch["y_true"].to(device).squeeze(1)
            mask = yt[:, 0] > 0
            if not mask.any(): continue
            pred = model(di)[mask]; true = yt[mask]
            total += torch.sqrt(((pred - true)**2).sum(dim=1)).mean().item()
            count += 1
    return (total / count) if count > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split",              default="B2", choices=["A","B","B2"])
    p.add_argument("--epochs",             type=int,   default=900)
    p.add_argument("--lr",                 type=float, default=1e-4)
    p.add_argument("--batch_size",         type=int,   default=8)
    p.add_argument("--seed",               type=int,   default=0)
    p.add_argument("--inv_hidden_dim",     type=int,   default=256)
    p.add_argument("--fwd_hidden_dim",     type=int,   default=128)
    p.add_argument("--num_gnn_proc_layers",type=int,   default=4)
    p.add_argument("--gat_heads",          type=int,   default=16)
    p.add_argument("--num_interaction_layers", type=int, default=3)
    p.add_argument("--val_every",          type=int,   default=10)
    # Dummy args so run_all can pass them without error
    p.add_argument("--inv_pretrain_epochs",type=int,   default=None)
    p.add_argument("--fwd_pretrain_epochs",type=int,   default=None)
    p.add_argument("--warmup",             type=int,   default=0)
    p.add_argument("--max_lambda",         type=float, default=0.0)
    p.add_argument("--mu",                 type=float, default=0.0)
    p.add_argument("--alpha",              type=float, default=0.0)
    p.add_argument("--ckpt_alpha",         type=float, default=0.0)
    p.add_argument("--lr_phase2",          type=float, default=None)
    args = p.parse_args()

    # inv_pretrain_epochs overrides epochs if passed (for run_all compatibility)
    if args.inv_pretrain_epochs is not None:
        args.epochs = args.inv_pretrain_epochs

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  {LABEL} | split={args.split} seed={args.seed}")
    print(f"  epochs={args.epochs} lr={args.lr}")
    print(f"{'='*65}", flush=True)

    with open("data/processed/ogw_data.pkl","rb") as f:
        raw = pickle.load(f)
    train_ids, val_ids, test_ids = get_train_val_test_ids(
        args.split, list(raw.keys()), seed=args.seed)

    stats     = build_all_stats(raw, train_ids)
    norm_data = stats["normalized_data_map"]
    ds_kw = dict(
        inv_static_edge_index           = stats["k12_edge_index"],
        inv_edge_feature_col_idxs       = stats["inv_edge_feature_col_idxs"],
        fwd_propagation_col_idxs        = stats["propagation_col_idxs"],
        fixed_fft_bin_indices           = stats["fixed_fft_bin_indices"],
        amp_means                       = stats["amp_means"],
        amp_stds                        = stats["amp_stds"],
        lookback_fft                    = stats["lookback_fft"],
        average_baseline_energy_profile = stats["average_baseline_energy_profile"],
        global_max_delta_e              = stats["global_max_delta_e"],
    )
    tr = DataLoader(CoupledModelDataset(norm_data, train_ids, **ds_kw),
                    batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(CoupledModelDataset(norm_data, val_ids,   **ds_kw),
                    batch_size=args.batch_size, shuffle=False)
    te = DataLoader(CoupledModelDataset(norm_data, test_ids,  **ds_kw),
                    batch_size=args.batch_size, shuffle=False)

    n_prop    = len(stats["propagation_pair_indices"])
    inv_model = GNN_inv_HierarchicalAttention(
        hidden_dim=args.inv_hidden_dim, raw_node_feat_dim=2,
        num_attention_freqs=N_ATTENTION_FREQS,
        num_gnn_proc_layers=args.num_gnn_proc_layers,
        gat_attention_heads=args.gat_heads,
        decoder_mlp_hidden_dim=args.inv_hidden_dim,
        final_output_dim=2, decoder_pooling_type="max",
    ).to(device)
    # Forward model instantiated but not used — needed for checkpoint format compatibility
    fwd_model = DirectPathAttenuationGNN(
        raw_node_feat_dim=2, physical_edge_feat_dim=6,
        hidden_dim=args.fwd_hidden_dim, num_propagation_pairs=n_prop,
        num_interaction_layers=args.num_interaction_layers,
    ).to(device)

    ckpt_path = checkpoint_path(args.split, LABEL, args.seed)
    logger    = EpochLogger(args.split, LABEL, args.seed)
    crit      = nn.MSELoss()
    opt       = optim.Adam(inv_model.parameters(), lr=args.lr)
    sch       = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.8, patience=20)

    best_val_mae  = float("inf")
    best_test_mae = float("nan")
    best_test_mse = float("nan")

    for ep in range(1, args.epochs + 1):
        inv_model.train(); tl = 0.0
        for batch in tqdm(tr, leave=False, disable=_TQDM_DISABLE):
            opt.zero_grad()
            di = batch["data_inv"].to(device)
            yt = batch["y_true"].to(device).squeeze(1)
            loss = crit(inv_model(di), yt)
            loss.backward(); opt.step()
            tl += loss.item() * di.num_graphs
        tl /= len(tr.dataset)

        if ep % args.val_every == 0 or ep in (1, args.epochs):
            vm  = euclidean_mae(inv_model, vl, device)
            vmm = vm * PLATE_MM
            sch.step(vm)
            tm, tmm = evaluate_test(inv_model, te, device)
            logger.log(ep, "inv", tl, vm, vmm, tm, tmm,
                       lr=opt.param_groups[0]["lr"])
            print(f"[{LABEL}] Ep {ep:04d} │ Loss={tl:.5f} │ "
                  f"ValMAE={vmm:.1f}mm │ TestMAE={tmm:.1f}mm │ "
                  f"LR={opt.param_groups[0]['lr']:.2e}", flush=True)

            if vmm < best_val_mae:
                best_val_mae  = vmm
                best_test_mae = tmm
                best_test_mse = tm
                save_checkpoint(ckpt_path, config=vars(args),
                                test_loss=tm, val_loss=vmm,
                                inv_model=inv_model.state_dict(),
                                fwd_model=fwd_model.state_dict())
                print(f"  ★ Ep {ep:04d} | ValMAE={vmm:.1f}mm → "
                      f"TestMAE={tmm:.1f}mm [saved]", flush=True)

    logger.close()
    log_result(args.split, f"{LABEL} (seed={args.seed})", best_test_mse)
    log_mae_result(args.split, LABEL, args.seed, best_val_mae, best_test_mae)
    print(f"\n[DONE] {LABEL} | best_val_mae={best_val_mae:.1f}mm | "
          f"test_mae={best_test_mae:.1f}mm | test_mse={best_test_mse:.5f}",
          flush=True)


if __name__ == "__main__":
    main()