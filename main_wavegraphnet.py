"""
WaveGraphNet — faithful implementation of the coupling-models.ipynb notebook.

Training protocol (exactly as in the notebook):

  Phase 1  (--fwd_pretrain_epochs, default 500):
    Pre-train forward model with ground-truth coordinates.
    Loss: focus-weighted MSE on delta-E.
    Optimizer: opt_fwd (fwd params only).
    inv_model is untouched.

  Phase 2  (--epochs, default 500):
    Train inv_model with combined loss.
    gnn_inv.train(), gnn_fwd.eval() — fwd FROZEN throughout.
    optimizer_inv contains ONLY inv_model parameters.
    Loss: Lloc + lambda(epoch) * Lfwd
    Gradient from Lfwd flows through predicted_locs INTO inv_model
    (NO detach — this is the physics consistency signal).
    lambda anneals 0 → max_lambda after warmup_epochs.
    Scheduler steps on eval MAE.
    Best checkpoint by val MAE.

Note: INV_ONLY_EPOCHS=0 in the notebook — no separate inverse pre-training.
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
    """Notebook get_lambda: linear ramp 0 → max_lam after warmup."""
    if epoch < warmup: return 0.0
    dur = total - warmup
    return min((epoch - warmup) / dur * max_lam, max_lam) if dur > 0 else max_lam


def make_fwd_batch(data_inv, prop_ei, device):
    prop_ei = prop_ei.to(device)
    return Batch.from_data_list([
        PyGData(x=data_inv.x[data_inv.batch == i], edge_index=prop_ei)
        for i in range(data_inv.num_graphs)
    ]).to(device)


# ── Phase 1: forward pre-training ─────────────────────────────────────────────
def train_phase1_fwd(fwd_model, loader, opt_fwd, prop_ei, device):
    """Notebook train_phase1_attenuation: focus-weighted MSE with GT coords."""
    fwd_model.train()
    total = 0.0
    for batch in tqdm(loader, leave=False, disable=_TQDM_DISABLE):
        opt_fwd.zero_grad()
        di = batch["data_inv"].to(device)
        yt = batch["y_true"].to(device).squeeze(1)   # [B, 2] GT coords
        de = batch["delta_e_true"].to(device)
        gf = make_fwd_batch(di, prop_ei, device)
        w  = de + 0.01
        loss = (100.0 * w * (fwd_model(gf, yt) - de) ** 2).mean()
        loss.backward(); opt_fwd.step()
        total += loss.item() * di.num_graphs
    return total / len(loader.dataset)


# ── Phase 2: inverse training with frozen forward ─────────────────────────────
def train_phase2_inv(inv_model, fwd_model, loader, opt_inv,
                     loss_fn_inv, loss_fn_fwd, lambda_fwd, prop_ei, device):
    inv_model.train()
    fwd_model.eval()
    crit = nn.MSELoss()
    tot_inv = tot_fwd = 0.0

    for batch in tqdm(loader, leave=False, disable=_TQDM_DISABLE):
        opt_inv.zero_grad()
        di = batch["data_inv"].to(device)
        yt = batch["y_true"].to(device).squeeze(1)
        de = batch["delta_e_true"].to(device)

        predicted_locs = inv_model(di)

        dmg_mask = yt[:, 0] > 0
        ud_mask  = ~dmg_mask

        # Damaged: exact coordinate regression
        loss_inv = crit(predicted_locs[dmg_mask], yt[dmg_mask]) \
                   if dmg_mask.any() else torch.tensor(0.0, device=device)

        # Undamaged: soft plate-exit penalty (not exact sentinel)
        if ud_mask.any():
            px = torch.clamp(predicted_locs[ud_mask, 0], min=0.0)
            py = torch.clamp(predicted_locs[ud_mask, 1], min=0.0)
            loss_inv = loss_inv + (px**2 + py**2).mean()

        # Forward consistency on damaged only
        loss_fwd = torch.tensor(0.0, device=device)
        if lambda_fwd > 0 and dmg_mask.any():
            gf = make_fwd_batch(di, prop_ei, device)
            loss_fwd = loss_fn_fwd(
                fwd_model(gf, predicted_locs)[dmg_mask], de[dmg_mask])

        (loss_inv + lambda_fwd * loss_fwd).backward()
        opt_inv.step()

        n = di.num_graphs
        tot_inv += loss_inv.item() * n
        tot_fwd += loss_fwd.item() * n

    n = len(loader.dataset)
    return tot_inv / n, tot_fwd / n


def euclidean_mae(model, loader, device):
    model.eval(); total = 0.0; count = 0
    with torch.no_grad():
        for batch in loader:
            di   = batch["data_inv"].to(device)
            yt   = batch["y_true"].to(device).squeeze(1)
            mask = yt[:, 0] > 0
            if not mask.any(): continue
            pred  = model(di)[mask]; true = yt[mask]
            total += torch.sqrt(((pred - true) ** 2).sum(dim=1)).mean().item()
            count += 1
    return (total / count) if count > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split",               default="A", choices=["A", "B"])
    p.add_argument("--mode",                default="coupled",
                   choices=["coupled", "inverse_only"])
    p.add_argument("--fwd_pretrain_epochs", type=int,  default=500,
                   help="Phase 1: forward pre-training (notebook EPOCHS_PHASE_1)")
    p.add_argument("--epochs",              type=int,  default=500,
                   help="Phase 2: inv training with fwd guidance (notebook EPOCHS_PHASE_2)")
    p.add_argument("--fwd_epochs",          type=int,  default=None)
    p.add_argument("--inv_pretrain_epochs", type=int,  default=0,
                   help="Stage 0: inverse-only pre-training epochs (0 = skip)")
    p.add_argument("--warmup",              type=int,  default=100,
                   help="notebook WARMUP_EPOCHS_PHASE_2")
    p.add_argument("--max_lambda",          type=float,default=100.0,
                   help="notebook MAX_LAMBDA_FWD")
    p.add_argument("--batch_size",          type=int,  default=8)
    p.add_argument("--lr",                  type=float,default=1e-4)
    p.add_argument("--lr_phase2",           type=float,default=None,
                   help="LR for Phase 2. Default=same as --lr. Set lower (e.g. 1e-5) "
                        "to avoid disrupting Stage 0 weights.")
    p.add_argument("--seed",                type=int,  default=42)
    p.add_argument("--inv_hidden_dim",      type=int,  default=256)
    p.add_argument("--fwd_hidden_dim",      type=int,  default=512)
    p.add_argument("--num_interaction_layers", type=int, default=8)
    p.add_argument("--gat_heads",           type=int,  default=16)
    p.add_argument("--num_gnn_proc_layers", type=int,  default=4)
    p.add_argument("--val_every",           type=int,  default=10)
    args = p.parse_args()

    if args.fwd_epochs is not None:
        args.fwd_pretrain_epochs = args.fwd_epochs

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/processed/ogw_data.pkl", "rb") as f:
        raw = pickle.load(f)

    train_ids, val_ids, test_ids = get_train_val_test_ids(
        args.split, list(raw.keys()), seed=args.seed)

    mode_label = "Coupled" if args.mode == "coupled" else "Inverse Only"
    full_label = f"WaveGraphNet ({mode_label})"
    print(f"\n{'='*65}")
    print(f"  {full_label} | split={args.split} seed={args.seed}")
    print(f"  Stage0(inv)={args.inv_pretrain_epochs}ep | "
          f"Phase1(fwd)={args.fwd_pretrain_epochs}ep | "
          f"Phase2(coupled)={args.epochs}ep | "
          f"λ_max={args.max_lambda} warmup={args.warmup}")
    print(f"{'='*65}", flush=True)

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
    fwd_model = DirectPathAttenuationGNN(
        raw_node_feat_dim=2, physical_edge_feat_dim=6,
        hidden_dim=args.fwd_hidden_dim, num_propagation_pairs=n_prop,
        num_interaction_layers=args.num_interaction_layers,
    ).to(device)

    prop_ei   = stats["propagation_edge_index"]
    ckpt_path = checkpoint_path(args.split, full_label, args.seed)
    ckpt_fwd  = ckpt_path.replace(".pt", "_fwd_best.pt")
    logger    = EpochLogger(args.split, full_label, args.seed)

    loss_fn_inv = nn.MSELoss()
    loss_fn_fwd = nn.MSELoss()

    best_val_mae  = float("inf")
    best_test_mae = float("nan")
    best_test_mse = float("nan")

    # ── Stage 0: optional inverse pre-training ────────────────────────────────
    if args.inv_pretrain_epochs and args.inv_pretrain_epochs > 0:
        print(f"\n--- Stage 0: Inverse pre-training ({args.inv_pretrain_epochs} epochs) ---",
              flush=True)
        opt_inv_s0  = optim.Adam(inv_model.parameters(), lr=args.lr)
        sch_inv_s0  = optim.lr_scheduler.ReduceLROnPlateau(opt_inv_s0, factor=0.8, patience=20)
        best_s0_mae = float("inf")
        ckpt_s0     = ckpt_path.replace(".pt", "_stage0_inv_best.pt")

        for ep in range(1, args.inv_pretrain_epochs + 1):
            inv_model.train()
            total_s0 = 0.0
            for batch in tqdm(tr, leave=False, disable=_TQDM_DISABLE):
                opt_inv_s0.zero_grad()
                di = batch["data_inv"].to(device)
                yt = batch["y_true"].to(device).squeeze(1)
                loss = loss_fn_inv(inv_model(di), yt)
                loss.backward(); opt_inv_s0.step()
                total_s0 += loss.item() * di.num_graphs
            total_s0 /= len(tr.dataset)

            if ep % args.val_every == 0 or ep in (1, args.inv_pretrain_epochs):
                vm  = euclidean_mae(inv_model, vl, device)
                vmm = vm * PLATE_MM
                sch_inv_s0.step(vm)
                tm, tmm = evaluate_test(inv_model, te, device)
                logger.log(ep, "S0_inv", total_s0, vm, vmm, tm, tmm,
                           lr=opt_inv_s0.param_groups[0]["lr"])

                if vmm < best_val_mae:
                    best_val_mae  = vmm
                    best_test_mae = tmm
                    best_test_mse = tm
                    save_checkpoint(ckpt_path, config=vars(args),
                                    test_loss=tm, val_loss=vmm,
                                    inv_model=inv_model.state_dict(),
                                    fwd_model=fwd_model.state_dict())
                    print(f"  ★ [S0] Ep {ep:04d} | ValMAE={vmm:.1f}mm → "
                          f"TestMAE={tmm:.1f}mm [saved]", flush=True)

                if vmm < best_s0_mae:
                    best_s0_mae = vmm
                    torch.save(inv_model.state_dict(), ckpt_s0)

        inv_model.load_state_dict(torch.load(ckpt_s0, map_location=device))
        print(f"  Stage 0 done. Best val MAE={best_s0_mae:.1f}mm → loaded.", flush=True)

    # ── Phase 1: forward pre-training ─────────────────────────────────────────
    if args.mode == "coupled" and args.fwd_pretrain_epochs > 0:
        print(f"\n--- Phase 1: Forward pre-training ({args.fwd_pretrain_epochs} epochs) ---",
              flush=True)
        opt_fwd = optim.Adam(fwd_model.parameters(), lr=args.lr)
        sch_fwd = optim.lr_scheduler.ReduceLROnPlateau(opt_fwd, factor=0.8, patience=20)
        best_fwd_loss = float("inf")

        for ep in range(1, args.fwd_pretrain_epochs + 1):
            fl = train_phase1_fwd(fwd_model, tr, opt_fwd, prop_ei, device)
            sch_fwd.step(fl)
            if ep % args.val_every == 0 or ep in (1, args.fwd_pretrain_epochs):
                print(f"  [P1 Fwd] Ep {ep:04d} | Loss={fl:.6f} "
                      f"| LR={opt_fwd.param_groups[0]['lr']:.2e}", flush=True)
            if fl < best_fwd_loss:
                best_fwd_loss = fl
                torch.save(fwd_model.state_dict(), ckpt_fwd)

        fwd_model.load_state_dict(torch.load(ckpt_fwd, map_location=device))
        print(f"  Phase 1 done. Best fwd loss={best_fwd_loss:.6f}", flush=True)

    # ── Phase 2: inv training with fwd guidance (fwd frozen) ──────────────────
    print(f"\n--- Phase 2: {mode_label} ({args.epochs} epochs) "
          f"[fwd frozen, only opt_inv updated] ---", flush=True)

    # ONLY inv_model parameters — this is what makes fwd frozen
    _lr_p2  = args.lr_phase2 if args.lr_phase2 is not None else args.lr
    opt_inv = optim.Adam(inv_model.parameters(), lr=_lr_p2)
    sch_inv = optim.lr_scheduler.ReduceLROnPlateau(opt_inv, factor=0.8, patience=20)

    # Reset best_val_mae for Phase 2 (Stage 0 metric is incomparable)
    if args.inv_pretrain_epochs and args.inv_pretrain_epochs > 0:
        best_val_mae  = float("inf")
        best_test_mae = float("nan")
        best_test_mse = float("nan")

    for epoch in range(1, args.epochs + 1):
        lam = 1.0#get_lambda(epoch - 1, args.epochs, args.warmup, args.max_lambda) \
              #if args.mode == "coupled" else 0.0

        l_inv, l_fwd = train_phase2_inv(
            inv_model, fwd_model, tr, opt_inv,
            loss_fn_inv, loss_fn_fwd, lam, prop_ei, device)

        if epoch % args.val_every == 0 or epoch in (1, args.epochs):
            vm  = euclidean_mae(inv_model, vl, device)
            vmm = vm * PLATE_MM
            sch_inv.step(vm)
            tm, tmm = evaluate_test(inv_model, te, device)
            logger.log(epoch, "P2_inv", l_inv, vm, vmm, tm, tmm,
                       lambda_fwd=lam, lr=opt_inv.param_groups[0]["lr"])

            if vmm < best_val_mae:
                best_val_mae  = vmm
                best_test_mae = tmm
                best_test_mse = tm
                save_checkpoint(ckpt_path, config=vars(args),
                                test_loss=tm, val_loss=vmm,
                                inv_model=inv_model.state_dict(),
                                fwd_model=fwd_model.state_dict())
                print(f"  ★ Ep {epoch:04d} | ValMAE={vmm:.1f}mm → "
                      f"TestMAE={tmm:.1f}mm TestMSE={tm:.5f} [saved]", flush=True)

    logger.close()
    log_result(args.split, f"{full_label} (seed={args.seed})", best_test_mse)
    log_mae_result(args.split, full_label, args.seed, best_val_mae, best_test_mae)
    print(f"\n[DONE] {full_label} | best_val_mae={best_val_mae:.1f}mm | "
          f"test_mae={best_test_mae:.1f}mm | test_mse={best_test_mse:.5f}", flush=True)


if __name__ == "__main__":
    main()