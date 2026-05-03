"""
WaveGraphNet — standard training script.

Implements the EXACT 3-stage training from the paper (Table 2 / Section 4.4):

  Stage I   (--inv_pretrain_epochs):
    Inverse branch trained alone.
    Optimizer: inv_model only.
    Loss: Lloc = MSE(pred_coords, true_coords)

  Stage II  (--fwd_pretrain_epochs):
    Forward branch trained alone with GROUND-TRUTH coordinates.
    Optimizer: fwd_model only.
    Loss: Lfwd (focus-weighted MSE on delta-E)

  Stage III (--epochs):
    BOTH branches jointly optimized together.
    Optimizer: inv_model + fwd_model parameters TOGETHER.
    Loss: Ltotal = Lloc + lambda(epoch) * Lfwd
    Lambda anneals 0 → max_lambda over warmup epochs.
    Scheduler steps on val Euclidean MAE.
    Best checkpoint saved by val MAE.

This is the correct implementation. The previous version had Stage I and II
swapped, and Stage III was never done (forward was frozen in the "coupled" phase
so only inverse was updated, making Coupled == Inverse Only with a noisy extra loss).
"""
import argparse, sys, random, pickle, os
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
    """Linear annealing 0 → max_lam after warmup epochs."""
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


# ── Stage I: inverse pre-training ─────────────────────────────────────────────
def train_stage1_inv(inv_model, loader, opt_inv, device) -> float:
    """
    Paper Stage I: train inverse branch alone.
    Loss = MSE(pred_coords, true_coords)
    """
    inv_model.train()
    crit = nn.MSELoss()
    total = 0.0
    for batch in tqdm(loader, leave=False, disable=_TQDM_DISABLE):
        opt_inv.zero_grad()
        di = batch["data_inv"].to(device)
        yt = batch["y_true"].to(device).squeeze(1)   # [B, 2]
        loss = crit(inv_model(di), yt)
        loss.backward(); opt_inv.step()
        total += loss.item() * di.num_graphs
    return total / len(loader.dataset)


# ── Stage II: forward pre-training ────────────────────────────────────────────
def train_stage2_fwd(fwd_model, loader, opt_fwd, prop_ei, device) -> float:
    """
    Paper Stage II: train forward branch alone with ground-truth coordinates.
    Loss = focus-weighted MSE on delta-E.
    """
    fwd_model.train()
    total = 0.0
    for batch in tqdm(loader, leave=False, disable=_TQDM_DISABLE):
        opt_fwd.zero_grad()
        di = batch["data_inv"].to(device)
        yt = batch["y_true"].to(device).squeeze(1)   # [B, 2] — GT coords for fwd
        de = batch["delta_e_true"].to(device)         # [B, 36]
        gf = make_fwd_batch(di, prop_ei, device)
        pd = fwd_model(gf, yt)                        # [B, 36]
        w  = de + 0.01
        loss = (100.0 * w * (pd - de) ** 2).mean()
        loss.backward(); opt_fwd.step()
        total += loss.item() * di.num_graphs
    return total / len(loader.dataset)


# ── Stage III: joint coupled training ─────────────────────────────────────────
def train_stage3_joint(inv_model, fwd_model, loader, opt_joint, lam,
                       prop_ei, device) -> tuple[float, float]:
    """
    Paper Stage III: BOTH branches optimized jointly.
    Inverse predicts coords → forward receives predicted coords → joint loss.
    optimizer_joint contains parameters from BOTH inv_model AND fwd_model.
    Loss: Ltotal = Lloc + lambda * Lfwd
    """
    inv_model.train()
    fwd_model.train()   # ← BOTH training, unlike the old frozen version
    crit    = nn.MSELoss()
    tot_inv = tot_fwd = 0.0

    for batch in tqdm(loader, leave=False, disable=_TQDM_DISABLE):
        opt_joint.zero_grad()
        di = batch["data_inv"].to(device)
        yt = batch["y_true"].to(device).squeeze(1)   # [B, 2]
        de = batch["delta_e_true"].to(device)         # [B, 36]

        pc    = inv_model(di)                         # [B, 2]  predicted coords
        l_inv = crit(pc, yt)                          # localization loss

        gf    = make_fwd_batch(di, prop_ei, device)
        l_fwd = crit(fwd_model(gf, pc), de)           # forward consistency loss

        loss  = l_inv + lam * l_fwd
        loss.backward()
        # Clip gradients for both models
        torch.nn.utils.clip_grad_norm_(inv_model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
        opt_joint.step()

        n = di.num_graphs
        tot_inv += l_inv.item() * n
        tot_fwd += l_fwd.item() * n

    n = len(loader.dataset)
    return tot_inv / n, tot_fwd / n


# ── Evaluation: Euclidean MAE over damaged samples only ───────────────────────
def euclidean_mae(model, loader, device) -> float:
    """Returns normalised MAE (multiply by PLATE_MM for mm)."""
    model.eval()
    total = 0.0; count = 0
    with torch.no_grad():
        for batch in loader:
            di   = batch["data_inv"].to(device)
            yt   = batch["y_true"].to(device).squeeze(1)   # [B, 2]
            mask = yt[:, 0] > 0                             # damaged only
            if not mask.any():
                continue
            pred  = model(di)[mask]
            true  = yt[mask]
            total += torch.sqrt(((pred - true) ** 2).sum(dim=1)).mean().item()
            count += 1
    return (total / count) if count > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split",                  default="A", choices=["A", "B"])
    p.add_argument("--mode",                   default="coupled",
                   choices=["coupled", "inverse_only"])

    # Stage epoch counts (paper uses 500 for each)
    p.add_argument("--inv_pretrain_epochs", type=int,  default=500,
                   help="Stage I: inverse pre-training epochs")
    p.add_argument("--fwd_pretrain_epochs", type=int,  default=500,
                   help="Stage II: forward pre-training epochs")
    p.add_argument("--epochs",              type=int,  default=500,
                   help="Stage III: joint coupled training epochs")

    # CLI alias for run_all.py compatibility (--fwd_epochs maps to fwd_pretrain_epochs)
    p.add_argument("--fwd_epochs",          type=int,  default=None,
                   help="Alias for --fwd_pretrain_epochs (run_all.py compat)")

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

    # Handle alias
    if args.fwd_epochs is not None:
        args.fwd_pretrain_epochs = args.fwd_epochs

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/processed/ogw_data.pkl", "rb") as f:
        raw_data_map = pickle.load(f)

    train_ids, val_ids, test_ids = get_train_val_test_ids(
        args.split, list(raw_data_map.keys()), seed=args.seed)

    mode_label = "Coupled" if args.mode == "coupled" else "Inverse Only"
    full_label = f"WaveGraphNet ({mode_label})"
    print(f"\n{'='*65}")
    print(f"  {full_label} | split={args.split} seed={args.seed}")
    print(f"  Stage I={args.inv_pretrain_epochs}ep (inv) | "
          f"Stage II={args.fwd_pretrain_epochs}ep (fwd) | "
          f"Stage III={args.epochs}ep (joint) | λ_max={args.max_lambda}")
    print(f"{'='*65}", flush=True)

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
    # Keep intermediate best checkpoints for Stage I and II
    ckpt_s1   = ckpt_path.replace(".pt", "_stage1_inv_best.pt")
    ckpt_s2   = ckpt_path.replace(".pt", "_stage2_fwd_best.pt")
    logger    = EpochLogger(args.split, full_label, args.seed)

    best_val_mae  = float("inf")
    best_test_mae = float("nan")
    best_test_mse = float("nan")

    # ──────────────────────────────────────────────────────────────────────────
    # Stage I: Inverse pre-training
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  STAGE I: Inverse pre-training ({args.inv_pretrain_epochs} epochs)")
    print(f"{'─'*65}", flush=True)

    opt_inv_s1  = optim.Adam(inv_model.parameters(), lr=args.lr)
    sch_inv_s1  = optim.lr_scheduler.ReduceLROnPlateau(opt_inv_s1, factor=0.8, patience=20)
    best_s1_mae = float("inf")

    for ep in range(1, args.inv_pretrain_epochs + 1):
        il = train_stage1_inv(inv_model, tr, opt_inv_s1, device)

        if ep % args.val_every == 0 or ep in (1, args.inv_pretrain_epochs):
            vm  = euclidean_mae(inv_model, vl, device)
            vmm = vm * PLATE_MM
            sch_inv_s1.step(vm)
            tm, tmm = evaluate_test(inv_model, te, device)
            logger.log(ep, "S1_inv", il, vm, vmm, tm, tmm,
                       lr=opt_inv_s1.param_groups[0]["lr"])

            if vmm < best_s1_mae:
                best_s1_mae = vmm
                torch.save(inv_model.state_dict(), ckpt_s1)

            if vmm < best_val_mae:
                best_val_mae  = vmm
                best_test_mae = tmm
                best_test_mse = tm
                save_checkpoint(ckpt_path, config=vars(args), test_loss=tm,
                                val_loss=vmm, inv_model=inv_model.state_dict(),
                                fwd_model=fwd_model.state_dict())
                print(f"  ★ [S1] Ep {ep:04d} ValMAE={vmm:.1f}mm → "
                      f"TestMAE={tmm:.1f}mm [saved]", flush=True)

    # Load best Stage I inverse for Stage III initialization
    print(f"  Stage I best val MAE={best_s1_mae:.1f}mm → loading best inv.",
          flush=True)
    inv_model.load_state_dict(torch.load(ckpt_s1, map_location=device))

    # ──────────────────────────────────────────────────────────────────────────
    # Stage II: Forward pre-training (only if coupled mode)
    # ──────────────────────────────────────────────────────────────────────────
    if args.mode == "coupled" and args.fwd_pretrain_epochs > 0:
        print(f"\n{'─'*65}")
        print(f"  STAGE II: Forward pre-training ({args.fwd_pretrain_epochs} epochs)")
        print(f"{'─'*65}", flush=True)

        opt_fwd_s2  = optim.Adam(fwd_model.parameters(), lr=args.lr)
        sch_fwd_s2  = optim.lr_scheduler.ReduceLROnPlateau(opt_fwd_s2, factor=0.8, patience=20)
        best_s2_fwd = float("inf")

        for ep in range(1, args.fwd_pretrain_epochs + 1):
            fl = train_stage2_fwd(fwd_model, tr, opt_fwd_s2, prop_ei, device)
            sch_fwd_s2.step(fl)

            if ep % args.val_every == 0 or ep in (1, args.fwd_pretrain_epochs):
                print(f"  [S2 Fwd] Ep {ep:04d} | FwdLoss={fl:.6f} "
                      f"| LR={opt_fwd_s2.param_groups[0]['lr']:.2e}", flush=True)

            if fl < best_s2_fwd:
                best_s2_fwd = fl
                torch.save(fwd_model.state_dict(), ckpt_s2)

        print(f"  Stage II best fwd loss={best_s2_fwd:.6f} → loading best fwd.",
              flush=True)
        fwd_model.load_state_dict(torch.load(ckpt_s2, map_location=device))

    # ──────────────────────────────────────────────────────────────────────────
    # Stage III: Joint coupled training (BOTH models trained together)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  STAGE III: Joint {'coupled' if args.mode=='coupled' else 'inverse-only'} "
          f"training ({args.epochs} epochs)")
    print(f"{'─'*65}", flush=True)

    if args.mode == "coupled":
        # BOTH models in the optimizer — this is the key fix
        opt_joint = optim.Adam(
            list(inv_model.parameters()) + list(fwd_model.parameters()),
            lr=args.lr
        )
    else:
        # Inverse only (no forward loss)
        opt_joint = optim.Adam(inv_model.parameters(), lr=args.lr)

    sch_joint = optim.lr_scheduler.ReduceLROnPlateau(opt_joint, factor=0.8, patience=20)

    for epoch in range(1, args.epochs + 1):
        lam = get_lambda(epoch - 1, args.epochs, args.warmup, args.max_lambda) \
              if args.mode == "coupled" else 0.0

        l_inv, l_fwd = train_stage3_joint(
            inv_model, fwd_model, tr, opt_joint, lam, prop_ei, device)

        if epoch % args.val_every == 0 or epoch in (1, args.epochs):
            vm  = euclidean_mae(inv_model, vl, device)
            vmm = vm * PLATE_MM
            sch_joint.step(vm)

            tm, tmm = evaluate_test(inv_model, te, device)
            logger.log(epoch, "S3_jnt", l_inv, vm, vmm, tm, tmm,
                       lambda_fwd=lam, lr=opt_joint.param_groups[0]["lr"])

            if vmm < best_val_mae:
                best_val_mae  = vmm
                best_test_mae = tmm
                best_test_mse = tm
                save_checkpoint(ckpt_path, config=vars(args),
                                test_loss=tm, val_loss=vmm,
                                inv_model=inv_model.state_dict(),
                                fwd_model=fwd_model.state_dict())
                print(f"  ★ [S3] Ep {epoch:04d} | ValMAE={vmm:.1f}mm → "
                      f"TestMAE={tmm:.1f}mm TestMSE={tm:.5f} [saved]", flush=True)

    logger.close()
    log_result(args.split, f"{full_label} (seed={args.seed})", best_test_mse)
    log_mae_result(args.split, full_label, args.seed, best_val_mae, best_test_mae)
    print(f"\n[DONE] {full_label} | best_val_mae={best_val_mae:.1f}mm | "
          f"test_mae={best_test_mae:.1f}mm | test_mse={best_test_mse:.5f}", flush=True)


if __name__ == "__main__":
    main()