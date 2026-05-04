"""
WaveGraphNet Trial

Improvements over main_wavegraphnet.py (notebook faithful):
  1. Undamaged samples masked from Lfwd — baselines have no physics meaning
     for the forward model (sentinel coordinate [-0.001,-0.001] outside plate).
  2. Combined checkpoint score in Phase 2:
       score = val_MSE_loc + alpha * val_MSE_fwd
     This selects the checkpoint where the inverse is simultaneously good at
     localization AND physically consistent with the forward model.
  3. Consistent val MSE metric across Stage 0 and Phase 2.

Training protocol:
  Stage 0  (--inv_pretrain_epochs, default 0):
    Optional inverse pre-training. Checkpoint by val_MSE_loc only.

  Phase 1  (--fwd_pretrain_epochs, default 500):
    Forward pre-training with GT coordinates + focus loss. Identical to notebook.

  Phase 2  (--epochs, default 500):
    gnn_inv.train(), gnn_fwd.eval() — fwd FROZEN (critical: eval not train).
    optimizer_inv contains ONLY inv params.
    Loss = Lloc + lambda(epoch) * Lfwd_masked
    Checkpoint saved when (val_MSE_loc + alpha * val_MSE_fwd) improves.
    alpha controls how much forward consistency contributes to checkpoint selection.
"""
import argparse, sys, random, pickle, os
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as PyGData, Batch
from tqdm import tqdm

from utils.splits import get_train_val_test_ids
from utils.data_loader_1 import CoupledModelDataset
from utils.precompute import build_all_stats, N_ATTENTION_FREQS
from utils.checkpointer import save_checkpoint, checkpoint_path
from models.wavegraphnet_1 import GNN_inv_HierarchicalAttention, DirectPathAttenuationGNN
from utils.logger import log_result, log_mae_result, EpochLogger, evaluate_test

_TQDM_DISABLE = not sys.stdout.isatty()
PLATE_MM      = 500.0
LABEL         = "WaveGraphNet (Trial)"


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_lambda(epoch, total, warmup, max_lam):
    """Linear ramp 0 → max_lam after warmup."""
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
    """Focus-weighted MSE with GT coords — identical to notebook."""
    fwd_model.train(); total = 0.0
    for batch in tqdm(loader, leave=False, disable=_TQDM_DISABLE):
        opt_fwd.zero_grad()
        di = batch["data_inv"].to(device)
        yt = batch["y_true"].to(device).squeeze(1)
        de = batch["delta_e_true"].to(device)
        gf = make_fwd_batch(di, prop_ei, device)
        loss = (100.0 * (de + 0.01) * (fwd_model(gf, yt) - de) ** 2).mean()
        loss.backward(); opt_fwd.step()
        total += loss.item() * di.num_graphs
    return total / len(loader.dataset)


# ── Phase 2: coupled training (undamaged masking) ─────────────────────────────
def train_phase2_trial(inv_model, fwd_model, loader, opt_inv,
                       lam, prop_ei, device):
    """
    Fixed-lambda coupled training with undamaged masking.
    fwd_model.eval() — FROZEN throughout. Only opt_inv updated.
    Gradient from Lfwd flows through predicted_locs into inv_model (no detach).
    Lfwd computed on DAMAGED samples only.
    """
    inv_model.train()
    fwd_model.eval()   # FROZEN
    crit = nn.MSELoss()
    tot_inv = tot_fwd = 0.0

    for batch in tqdm(loader, leave=False, disable=_TQDM_DISABLE):
        opt_inv.zero_grad()
        di = batch["data_inv"].to(device)
        yt = batch["y_true"].to(device).squeeze(1)   # [B, 2]
        de = batch["delta_e_true"].to(device)         # [B, 36]

        pc    = inv_model(di)                         # [B, 2]
        l_inv = crit(pc, yt)

        if lam > 0:
            dmg_mask = yt[:, 0] > 0                  # damaged samples only
            if dmg_mask.any():
                gf    = make_fwd_batch(di, prop_ei, device)
                l_fwd = crit(fwd_model(gf, pc)[dmg_mask], de[dmg_mask])
                (l_inv + lam * l_fwd).backward()
                tot_fwd += l_fwd.item() * di.num_graphs
            else:
                l_inv.backward()
        else:
            l_inv.backward()

        opt_inv.step()
        tot_inv += l_inv.item() * di.num_graphs

    n = len(loader.dataset)
    return tot_inv / n, tot_fwd / n


# ── Validation metrics ─────────────────────────────────────────────────────────
def eval_val(inv_model, fwd_model, loader, prop_ei, device, alpha=0.0):
    """
    Computes three validation terms, each in [0,1] range, combined with equal weight.

    mse_loc  — MSE(pred_coords, true_coords) on damaged val samples only.
               Normalized coords [0,1]^2. Theoretical max = 1.0.

    undamaged_soft — Soft penalty for undamaged samples predicted inside the plate.
               For each undamaged sample: (clamp(px,0)^2 + clamp(py,0)^2) / 2
               = 0 if predicted outside plate (correct), up to 1.0 if at [1,1].
               Does NOT penalize the exact value — only whether the model
               correctly places predictions outside the plate domain.

    mse_fwd  — MSE(fwd(pc), delta_e_true) on damaged samples.
               delta_e normalized to [0,1]. Theoretical max = 1.0.
               Only computed when alpha > 0.

    score = (mse_loc + undamaged_soft + alpha * mse_fwd) / (2 + alpha)
    This gives true equal weighting when alpha=1: score = mean of all three terms.
    alpha=0: score = (mse_loc + undamaged_soft) / 2 — no forward term.

    Returns: mse_loc, undamaged_soft, mse_fwd, mae_mm, score
    """
    inv_model.eval(); fwd_model.eval()
    crit = nn.MSELoss()
    tot_loc = tot_fwd = tot_eucl = 0.0
    tot_ud  = 0.0    # undamaged soft penalty accumulator
    n_dmg   = 0      # number of batches with damaged samples
    n_ud    = 0      # total undamaged sample count

    with torch.no_grad():
        for batch in loader:
            di   = batch["data_inv"].to(device)
            yt   = batch["y_true"].to(device).squeeze(1)   # [B, 2]
            de   = batch["delta_e_true"].to(device)

            pc   = inv_model(di)                            # [B, 2]

            dmg_mask = yt[:, 0] > 0    # damaged
            ud_mask  = ~dmg_mask       # undamaged (baselines)

            # ── Localization MSE on damaged ───────────────────────────────
            if dmg_mask.any():
                tot_loc  += crit(pc[dmg_mask], yt[dmg_mask]).item()                             * dmg_mask.sum().item()
                tot_eucl += torch.sqrt(
                    ((pc[dmg_mask] - yt[dmg_mask])**2).sum(dim=1)
                ).mean().item()
                n_dmg += 1

                # Forward consistency on damaged only
                if alpha > 0:
                    gf = make_fwd_batch(di, prop_ei, device)
                    tot_fwd += crit(
                        fwd_model(gf, pc)[dmg_mask], de[dmg_mask]
                    ).item() * dmg_mask.sum().item()

            # ── Undamaged soft penalty ────────────────────────────────────
            if ud_mask.any():
                # Penalize only how far INSIDE the plate [0,1]^2 the prediction lands.
                # clamp(px, min=0): 0 if px<0 (outside), px if px>=0 (inside).
                # Divide by 2 (two coords) → range [0, 1] per sample.
                px = torch.clamp(pc[ud_mask, 0], min=0.0)
                py = torch.clamp(pc[ud_mask, 1], min=0.0)
                penalty = ((px**2 + py**2) / 2.0).sum().item()
                tot_ud  += penalty
                n_ud    += ud_mask.sum().item()

    # Normalize each term by its sample count
    total_dmg = sum(
        (b["y_true"].squeeze(1)[:, 0] > 0).sum().item() for b in loader
    )
    total_ud = sum(
        (b["y_true"].squeeze(1)[:, 0] <= 0).sum().item() for b in loader
    )
    total_dmg = max(total_dmg, 1)
    total_ud  = max(total_ud,  1)

    mse_loc        = tot_loc  / total_dmg
    mse_fwd        = tot_fwd  / total_dmg
    undamaged_soft = tot_ud   / total_ud
    mae_mm         = (tot_eucl / max(n_dmg, 1)) * PLATE_MM

    # Equal weighting: score = (mse_loc + undamaged_soft + alpha*mse_fwd) / (2+alpha)
    denom = 2.0 + alpha
    score = (mse_loc + undamaged_soft + alpha * mse_fwd) / denom

    return mse_loc, undamaged_soft, mse_fwd, mae_mm, score


def euclidean_mae(model, loader, device):
    """Legacy helper used by evaluate_test."""
    model.eval(); total = 0.0; count = 0
    with torch.no_grad():
        for batch in loader:
            di   = batch["data_inv"].to(device)
            yt   = batch["y_true"].to(device).squeeze(1)
            mask = yt[:, 0] > 0
            if not mask.any(): continue
            pred  = model(di)[mask]; true = yt[mask]
            total += torch.sqrt(((pred - true)**2).sum(dim=1)).mean().item()
            count += 1
    return (total / count) if count > 0 else float("nan")


def main():
    p = argparse.ArgumentParser(description="WaveGraphNet Trial")
    p.add_argument("--split",               default="A",  choices=["A","B"])
    p.add_argument("--mode",                default="coupled",
                   choices=["coupled","inverse_only"])
    p.add_argument("--inv_pretrain_epochs", type=int,   default=0)
    p.add_argument("--fwd_pretrain_epochs", type=int,   default=500)
    p.add_argument("--epochs",              type=int,   default=500)
    p.add_argument("--warmup",              type=int,   default=100,
                   help="Epochs with lambda=0 in Phase 2")
    p.add_argument("--max_lambda",          type=float, default=0.5,
                   help="Max forward consistency weight (ramps 0→max after warmup)")
    p.add_argument("--alpha",               type=float, default=1.0,
                   help="Weight of val_MSE_fwd in checkpoint score: "
                        "score = val_MSE_loc + alpha * val_MSE_fwd. "
                        "alpha=0 → checkpoint by loc only (same as main_wavegraphnet.py). "
                        "alpha=1 → equal weight.")
    p.add_argument("--lr",                  type=float, default=1e-4)
    p.add_argument("--lr_phase2",           type=float, default=1e-4,
                   help="LR for Phase 2 (use 1e-5 if Stage 0 is used)")
    p.add_argument("--batch_size",          type=int,   default=8)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--inv_hidden_dim",      type=int,   default=256)
    p.add_argument("--fwd_hidden_dim",      type=int,   default=512)
    p.add_argument("--num_interaction_layers", type=int, default=8)
    p.add_argument("--gat_heads",           type=int,   default=16)
    p.add_argument("--num_gnn_proc_layers", type=int,   default=4)
    p.add_argument("--val_every",           type=int,   default=10)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  {LABEL} | split={args.split} seed={args.seed}")
    print(f"  Stage0={args.inv_pretrain_epochs}ep | "
          f"Phase1={args.fwd_pretrain_epochs}ep | "
          f"Phase2={args.epochs}ep | warmup={args.warmup}")
    print(f"  mode={args.mode} | λ_max={args.max_lambda} | "
          f"ckpt_score=MSE_loc + {args.alpha}×MSE_fwd")
    print(f"{'='*65}\n", flush=True)

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
        fwd_prop_edge_index             = stats["propagation_edge_index"],  # ← ADD THIS
        geometric_sigma                 = 0.1,                              # ← AND THIS
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
    ckpt_path = checkpoint_path(args.split, LABEL, args.seed)
    ckpt_fwd  = ckpt_path.replace(".pt", "_fwd_best.pt")
    logger    = EpochLogger(args.split, LABEL, args.seed)

    best_score    = float("inf")   # unified metric across all stages
    best_test_mae = float("nan")
    best_test_mse = float("nan")

    # ── Stage 0: optional inverse pre-training ────────────────────────────────
    if args.inv_pretrain_epochs > 0:
        print(f"--- Stage 0: Inverse pre-training ({args.inv_pretrain_epochs} ep) ---",
              flush=True)
        crit_s0   = nn.MSELoss()
        opt_s0    = optim.Adam(inv_model.parameters(), lr=args.lr)
        sch_s0    = optim.lr_scheduler.ReduceLROnPlateau(opt_s0, factor=0.8, patience=20)
        best_s0   = float("inf")
        ckpt_s0   = ckpt_path.replace(".pt", "_s0_best.pt")

        for ep in range(1, args.inv_pretrain_epochs + 1):
            inv_model.train(); tl = 0.0
            for batch in tqdm(tr, leave=False, disable=_TQDM_DISABLE):
                opt_s0.zero_grad()
                di = batch["data_inv"].to(device)
                yt = batch["y_true"].to(device).squeeze(1)
                l  = crit_s0(inv_model(di), yt)
                l.backward(); opt_s0.step()
                tl += l.item() * di.num_graphs
            tl /= len(tr.dataset)

            if ep % args.val_every == 0 or ep in (1, args.inv_pretrain_epochs):
                # Stage 0: score = val_MSE_loc only (no fwd available yet)
                mse_loc, ud_soft, _, mae_mm, score = eval_val(
                    inv_model, fwd_model, vl, prop_ei, device, alpha=0.0)
                sch_s0.step(mse_loc)
                tm, tmm = evaluate_test(inv_model, te, device)
                logger.log(ep, "S0_inv", tl,
                           mse_loc, mae_mm, tm, tmm,
                           lr=opt_s0.param_groups[0]["lr"])

                # Stage 0: no final checkpoint saved — only track best weights
                # for loading at Phase 2 start
                if mse_loc < best_s0:
                    best_s0 = mse_loc
                    torch.save(inv_model.state_dict(), ckpt_s0)
                    print(f"  [S0] Ep {ep:04d} | "
                          f"ValMSE_loc={mse_loc:.5f} (MAE={mae_mm:.1f}mm) → "
                          f"TestMAE={tmm:.1f}mm [best S0]", flush=True)

        inv_model.load_state_dict(torch.load(ckpt_s0, map_location=device))
        print(f"  Stage 0 done. best_val_MSE_loc={best_s0:.5f} → loaded.", flush=True)

    # ── Phase 1: forward pre-training ─────────────────────────────────────────
    if args.mode == "coupled" and args.fwd_pretrain_epochs > 0:
        print(f"\n--- Phase 1: Forward pre-training ({args.fwd_pretrain_epochs} ep) ---",
              flush=True)
        opt_fwd = optim.Adam(fwd_model.parameters(), lr=args.lr)
        sch_fwd = optim.lr_scheduler.ReduceLROnPlateau(opt_fwd, factor=0.8, patience=20)
        best_fwd = float("inf")

        for ep in range(1, args.fwd_pretrain_epochs + 1):
            fl = train_phase1_fwd(fwd_model, tr, opt_fwd, prop_ei, device)
            sch_fwd.step(fl)
            if ep % args.val_every == 0 or ep in (1, args.fwd_pretrain_epochs):
                print(f"  [P1] Ep {ep:04d} | FwdLoss={fl:.6f} "
                      f"| LR={opt_fwd.param_groups[0]['lr']:.2e}", flush=True)
            if fl < best_fwd:
                best_fwd = fl
                torch.save(fwd_model.state_dict(), ckpt_fwd)

        fwd_model.load_state_dict(torch.load(ckpt_fwd, map_location=device))
        print(f"  Phase 1 done. Best fwd loss={best_fwd:.6f}", flush=True)

    # ── Phase 2: coupled training ──────────────────────────────────────────────
    alpha_p2 = args.alpha if args.mode == "coupled" else 0.0
    print(f"\n--- Phase 2: {args.mode} ({args.epochs} ep) "
          f"| ckpt_score = MSE_loc + {alpha_p2:.1f}×MSE_fwd ---", flush=True)

    # Reset checkpoint score — Phase 2 uses MSE_loc + alpha*MSE_fwd,
    # incomparable with Stage 0 score (MSE_loc only). Start fresh.
    best_score    = float("inf")
    best_test_mae = float("nan")
    best_test_mse = float("nan")
    _lr_p2  = args.lr_phase2
    opt_inv = optim.Adam(inv_model.parameters(), lr=_lr_p2)
    sch_inv = optim.lr_scheduler.ReduceLROnPlateau(opt_inv, factor=0.8, patience=20)

    for epoch in range(1, args.epochs + 1):
        lam = get_lambda(epoch - 1, args.epochs, args.warmup, args.max_lambda) \
              if args.mode == "coupled" else 0.0

        l_inv, l_fwd = train_phase2_trial(
            inv_model, fwd_model, tr, opt_inv, lam, prop_ei, device)

        if epoch % args.val_every == 0 or epoch in (1, args.epochs):
            mse_loc, ud_soft, mse_fwd, mae_mm, score = eval_val(
                inv_model, fwd_model, vl, prop_ei, device, alpha=alpha_p2)
            sch_inv.step(mse_loc)
            tm, tmm = evaluate_test(inv_model, te, device)

            logger.log(epoch, "P2_trl", l_inv, mse_loc, mae_mm, tm, tmm,
                       lambda_fwd=lam, lr=opt_inv.param_groups[0]["lr"])

            print(f"[{LABEL}] Ep {epoch:04d} │ P2 │ "
                  f"MSE_loc={mse_loc:.5f} UD_soft={ud_soft:.5f} "
                  f"MSE_fwd={mse_fwd:.5f} score={score:.5f} "
                  f"MAE={mae_mm:.1f}mm │ Test={tmm:.1f}mm │ λ={lam:.3f}", flush=True)

            if score < best_score:
                best_score    = score
                best_test_mae = tmm
                best_test_mse = tm
                save_checkpoint(ckpt_path, config=vars(args),
                                test_loss=tm, val_loss=score,
                                inv_model=inv_model.state_dict(),
                                fwd_model=fwd_model.state_dict())
                print(f"  ★ Ep {epoch:04d} | score={score:.5f} "
                      f"(loc={mse_loc:.5f} ud={ud_soft:.5f} fwd={mse_fwd:.5f}) → "
                      f"TestMAE={tmm:.1f}mm [saved]", flush=True)

    logger.close()
    log_result(args.split, f"{LABEL} (seed={args.seed})", best_test_mse)
    log_mae_result(args.split, LABEL, args.seed,
                   best_score * PLATE_MM, best_test_mae)
    print(f"\n[DONE] {LABEL} | best_score={best_score:.5f} | "
          f"test_mae={best_test_mae:.1f}mm | test_mse={best_test_mse:.5f}",
          flush=True)


if __name__ == "__main__":
    main()