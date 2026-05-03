"""
WaveGraphNet — standard training script.

Mirrors coupling-models.ipynb exactly:
  Phase 1: Pre-train forward model only  (fwd_epochs)
  Phase 2: Train inverse only, forward FROZEN  (epochs)
            Loss = L_inv + lambda(epoch) * L_fwd
            Lambda anneals 0 → max_lambda after warmup epochs.
            Scheduler steps on val Euclidean MAE (normalised coords).
            Best checkpoint saved by val MAE.
"""
import argparse, sys, random, pickle
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as PyGData, Batch
from tqdm import tqdm

from utils.splits import get_train_val_test_ids
from utils.data_loader import CoupledModelDataset
from utils.precompute import build_all_stats, N_ATTENTION_FREQS
from utils.checkpointer import save_checkpoint, checkpoint_path
from models.wavegraphnet import GNN_inv_HierarchicalAttention, DirectPathAttenuationGNN
from utils.logger import log_result, log_mae_result, EpochLogger, evaluate_test

_TQDM_DISABLE = not sys.stdout.isatty()
PLATE_MM      = 500.0


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_lambda(epoch, total, warmup, max_lam):
    """Linear annealing 0 → max_lam after warmup (notebook Cell 44)."""
    if epoch < warmup:
        return 0.0
    dur = total - warmup
    return min((epoch - warmup) / dur * max_lam, max_lam) if dur > 0 else max_lam


def make_fwd_batch(data_inv: Batch, prop_ei: torch.Tensor, device) -> Batch:
    prop_ei = prop_ei.to(device)
    return Batch.from_data_list([
        PyGData(x=data_inv.x[data_inv.batch == i], edge_index=prop_ei)
        for i in range(data_inv.num_graphs)
    ]).to(device)


# ── Phase 1: forward pre-training ─────────────────────────────────────────────
def train_phase1(fwd_model, loader, opt, prop_ei, device):
    """Focus-weighted MSE on delta-E (notebook Cell 45)."""
    fwd_model.train()
    total = 0.0
    for batch in tqdm(loader, leave=False, disable=_TQDM_DISABLE):
        opt.zero_grad()
        di = batch["data_inv"].to(device)
        # FIX: squeeze(1) → [B, 2] not [B, 1, 2]
        yt = batch["y_true"].to(device).squeeze(1)   # [B, 2]
        de = batch["delta_e_true"].to(device)         # [B, 36]
        gf = make_fwd_batch(di, prop_ei, device)
        pd = fwd_model(gf, yt)                        # [B, 36]
        w  = de + 0.01
        loss = (100.0 * w * (pd - de) ** 2).mean()
        loss.backward(); opt.step()
        total += loss.item() * di.num_graphs
    return total / len(loader.dataset)


# ── Phase 2: inverse training with frozen forward ─────────────────────────────
def train_phase2(inv_model, fwd_model, loader, opt_inv, lam, prop_ei, device):
    """
    Notebook Cell 50 exactly:
    - inv_model.train(), fwd_model.eval() (frozen, not in optimizer)
    - optimizer_inv contains ONLY inv_model parameters
    """
    inv_model.train()
    fwd_model.eval()    # frozen
    crit    = nn.MSELoss()
    tot_inv = tot_fwd = 0.0

    for batch in tqdm(loader, leave=False, disable=_TQDM_DISABLE):
        opt_inv.zero_grad()
        di = batch["data_inv"].to(device)
        # FIX: squeeze(1) → [B, 2]
        yt = batch["y_true"].to(device).squeeze(1)   # [B, 2]
        de = batch["delta_e_true"].to(device)         # [B, 36]

        pc    = inv_model(di)                         # [B, 2]
        l_inv = crit(pc, yt)                          # scalar

        gf    = make_fwd_batch(di, prop_ei, device)
        l_fwd = crit(fwd_model(gf, pc), de)           # scalar

        (l_inv + lam * l_fwd).backward()
        opt_inv.step()

        n = di.num_graphs
        tot_inv += l_inv.item() * n
        tot_fwd += l_fwd.item() * n

    n = len(loader.dataset)
    return tot_inv / n, tot_fwd / n


# ── Evaluation: Euclidean MAE over damaged samples ────────────────────────────
def euclidean_mae(model, loader, device) -> float:
    """
    Matches notebook Cell 47:
      error = sqrt(sum_sq).mean() per batch → average over batches
    Returns normalised MAE (divide by PLATE_MM for mm).
    """
    model.eval()
    total = 0.0; count = 0
    with torch.no_grad():
        for batch in loader:
            di   = batch["data_inv"].to(device)
            # FIX: squeeze(1) → [B, 2]
            yt   = batch["y_true"].to(device).squeeze(1)   # [B, 2]
            mask = yt[:, 0] > 0                             # [B] bool
            if not mask.any():
                continue
            pred  = model(di)[mask]                         # [K, 2]
            true  = yt[mask]                                # [K, 2]
            total += torch.sqrt(((pred - true) ** 2).sum(dim=1)).mean().item()
            count += 1
    return (total / count) if count > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split",                  default="A", choices=["A", "B"])
    p.add_argument("--mode",                   default="coupled",
                   choices=["coupled", "inverse_only"])
    p.add_argument("--fwd_epochs",  type=int,  default=500)
    p.add_argument("--epochs",      type=int,  default=500)
    p.add_argument("--warmup",      type=int,  default=100)
    p.add_argument("--max_lambda",  type=float,default=100.0)
    p.add_argument("--batch_size",  type=int,  default=8)
    p.add_argument("--lr",          type=float,default=1e-4)
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--inv_hidden_dim",         type=int, default=256)
    p.add_argument("--fwd_hidden_dim",         type=int, default=512)
    p.add_argument("--num_interaction_layers", type=int, default=8)
    p.add_argument("--gat_heads",   type=int,  default=16)
    p.add_argument("--num_gnn_proc_layers",    type=int, default=4)
    p.add_argument("--val_every",   type=int,  default=10)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/processed/ogw_data.pkl", "rb") as f:
        raw_data_map = pickle.load(f)

    train_ids, val_ids, test_ids = get_train_val_test_ids(
        args.split, list(raw_data_map.keys()), seed=args.seed)

    mode_label = "Coupled" if args.mode == "coupled" else "Inverse Only"
    full_label = f"WaveGraphNet ({mode_label})"
    print(f"\n{'='*65}\n  {full_label} | split={args.split} seed={args.seed}\n{'='*65}",
          flush=True)

    stats     = build_all_stats(raw_data_map, train_ids)
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
        hidden_dim             = args.inv_hidden_dim,
        raw_node_feat_dim      = 2,
        num_attention_freqs    = N_ATTENTION_FREQS,
        num_gnn_proc_layers    = args.num_gnn_proc_layers,
        gat_attention_heads    = args.gat_heads,
        decoder_mlp_hidden_dim = args.inv_hidden_dim,
        final_output_dim       = 2,
        decoder_pooling_type   = "max",
    ).to(device)
    fwd_model = DirectPathAttenuationGNN(
        raw_node_feat_dim      = 2,
        physical_edge_feat_dim = 6,
        hidden_dim             = args.fwd_hidden_dim,
        num_propagation_pairs  = n_prop,
        num_interaction_layers = args.num_interaction_layers,
    ).to(device)

    prop_ei   = stats["propagation_edge_index"]
    ckpt_path = checkpoint_path(args.split, full_label, args.seed)
    logger    = EpochLogger(args.split, full_label, args.seed)

    # ── Phase 1: forward pre-training ─────────────────────────────────────────
    if args.mode == "coupled" and args.fwd_epochs > 0:
        print(f"\n--- Phase 1: Forward pre-training ({args.fwd_epochs} epochs) ---",
              flush=True)
        opt_f         = optim.Adam(fwd_model.parameters(), lr=args.lr)
        sch_f         = optim.lr_scheduler.ReduceLROnPlateau(opt_f, factor=0.8, patience=20)
        best_fwd_loss = float("inf")
        best_fwd_path = ckpt_path.replace(".pt", "_fwd_best.pt")

        for ep in range(1, args.fwd_epochs + 1):
            fl = train_phase1(fwd_model, tr, opt_f, prop_ei, device)
            sch_f.step(fl)
            if ep % args.val_every == 0 or ep in (1, args.fwd_epochs):
                print(f"  [Fwd P1] Ep {ep:04d} | FwdLoss={fl:.6f} "
                      f"| LR={opt_f.param_groups[0]['lr']:.2e}", flush=True)
            if fl < best_fwd_loss:
                best_fwd_loss = fl
                torch.save(fwd_model.state_dict(), best_fwd_path)

        print(f"  Loading best P1 forward model (loss={best_fwd_loss:.6f})")
        fwd_model.load_state_dict(torch.load(best_fwd_path, map_location=device))

    # ── Phase 2: inverse training (forward frozen) ────────────────────────────
    print(f"\n--- Phase 2: {mode_label} ({args.epochs} epochs) ---", flush=True)
    # ONLY inv_model params — forward is frozen and stays frozen
    opt_inv = optim.Adam(inv_model.parameters(), lr=args.lr)
    sch_inv = optim.lr_scheduler.ReduceLROnPlateau(opt_inv, factor=0.8, patience=20)

    best_val_mae  = float("inf")
    best_test_mae = float("nan")
    best_test_mse = float("nan")

    for epoch in range(1, args.epochs + 1):
        lam = get_lambda(epoch - 1, args.epochs, args.warmup, args.max_lambda) \
              if args.mode == "coupled" else 0.0

        l_inv, l_fwd = train_phase2(
            inv_model, fwd_model, tr, opt_inv, lam, prop_ei, device)

        if epoch % args.val_every == 0 or epoch in (1, args.epochs):
            val_mae_norm = euclidean_mae(inv_model, vl, device)
            val_mae_mm   = val_mae_norm * PLATE_MM
            # Scheduler steps on val MAE (matches notebook Cell 57)
            sch_inv.step(val_mae_norm)

            test_mse, test_mae_mm = evaluate_test(inv_model, te, device)
            logger.log(epoch, "train", l_inv,
                       val_mae_norm, val_mae_mm,
                       test_mse, test_mae_mm,
                       lambda_fwd=lam, lr=opt_inv.param_groups[0]["lr"])

            if val_mae_mm < best_val_mae:
                best_val_mae  = val_mae_mm
                best_test_mae = test_mae_mm
                best_test_mse = test_mse
                save_checkpoint(ckpt_path, config=vars(args),
                                test_loss=test_mse, val_loss=val_mae_mm,
                                inv_model=inv_model.state_dict(),
                                fwd_model=fwd_model.state_dict())
                print(f"  ★ Ep {epoch:04d} | ValMAE={val_mae_mm:.1f}mm → "
                      f"TestMAE={test_mae_mm:.1f}mm TestMSE={test_mse:.5f} [saved]",
                      flush=True)

    logger.close()
    log_result(args.split, f"{full_label} (seed={args.seed})", best_test_mse)
    log_mae_result(args.split, full_label, args.seed, best_val_mae, best_test_mae)
    print(f"\n[DONE] {full_label} | best_val_mae={best_val_mae:.1f}mm | "
          f"test_mae={best_test_mae:.1f}mm | test_mse={best_test_mse:.5f}", flush=True)


if __name__ == "__main__":
    main()