"""
WaveGraphNet — TRIAL alternating-curriculum training script.

Motivation
----------
The standard script (main_wavegraphnet.py) permanently freezes the forward
model after Phase 1.  As the inverse branch improves over 500 epochs, the
forward model's consistency signal becomes progressively misaligned because
it was trained with ground-truth coordinates, not the (imperfect but
improving) predicted coordinates it will receive during Phase 2.

This script replaces Phase 2 with a three-stage alternating curriculum:

  Stage 1  (stage1_epochs)   Forward pre-training with GT coordinates.
                              Focus-weighted MSE: L = mean(100*(de+0.01)*(ŷ-y)²)

  Stage 2  (stage2_inv_epochs)  Inverse warmup — no forward consistency.
                              L = MSE(pred_coords, true_coords)
                              Lets the inverse branch establish a reasonable
                              initialization before the forward signal kicks in.

  Stage 3  (stage3_cycles × cycle_len)  Alternating coupled training:
    ┌ INV phase (inv_per_cycle epochs per cycle)
    │   inv_model.train(), fwd_model.eval() (frozen teacher)
    │   L = L_inv  +  lambda(t) × L_fwd
    │   Only opt_inv updated.
    └ FWD-REFINE phase (fwd_per_cycle epochs per cycle)
        fwd_model.train(), inv_model.eval() (frozen)
        Coords = inv_model(batch)  [no_grad — just pseudo-labels]
        L = focus_loss(fwd_model(predicted_coords), true_delta_e)
        Only opt_fwd updated.

  This creates a progressive tightening loop:
    better inverse → better pseudo-labels for forward fine-tuning
    better forward → more accurate consistency signal for inverse
    ... repeat each cycle

Hyperparameters (all exposed as CLI flags):
  --stage1_epochs     forward pre-training epochs        (default 500)
  --stage2_inv_epochs inverse warmup epochs              (default 100)
  --stage3_cycles     number of alternating cycles       (default 8)
  --inv_per_cycle     inverse training epochs per cycle  (default 50)
  --fwd_per_cycle     forward fine-tuning epochs/cycle   (default 20)
  --warmup_cycles     lambda warmup (in stage-3 CYCLES)  (default 2)
  --max_lambda        max forward-consistency weight     (default 100)
  --lr                learning rate for all optimizers   (default 1e-4)
  --batch_size                                           (default 8)
  --val_every         val checkpoint every N epochs      (default 10)

All other fixes included:
  ✓ y_true.squeeze(1) in every train and eval function
  ✓ sorted() propagation edge order
  ✓ normalized_data_map used (time-domain preprocessing)
  ✓ Scheduler steps on val Euclidean MAE (not train MSE)
  ✓ Best checkpoint by val MAE; test MAE reported from best-val epoch
  ✓ Per-seed MAE logged to results_mae.json for mean±std aggregation
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
LABEL         = "WaveGraphNet (Trial)"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def make_fwd_batch(data_inv: Batch, prop_ei: torch.Tensor, device) -> Batch:
    """Build a fresh forward-graph Batch sharing prop_ei across all graphs."""
    prop_ei = prop_ei.to(device)
    return Batch.from_data_list([
        PyGData(x=data_inv.x[data_inv.batch == i], edge_index=prop_ei)
        for i in range(data_inv.num_graphs)
    ]).to(device)


def get_lambda_by_cycle(
    cycle: int, total_cycles: int, warmup_cycles: int, max_lam: float
) -> float:
    """Linear annealing over stage-3 cycles (same shape as notebook's epoch-wise)."""
    if cycle < warmup_cycles:
        return 0.0
    dur = total_cycles - warmup_cycles
    if dur <= 0:
        return max_lam
    return min((cycle - warmup_cycles) / dur * max_lam, max_lam)


# ─────────────────────────────────────────────────────────────────────────────
# Training steps
# ─────────────────────────────────────────────────────────────────────────────

def _focus_loss(pred_de: torch.Tensor, true_de: torch.Tensor) -> torch.Tensor:
    """Weighted MSE that up-weights high-energy paths (from notebook Cell 45)."""
    w = true_de + 0.01
    return (100.0 * w * (pred_de - true_de) ** 2).mean()


def train_stage1_fwd(fwd_model, loader, opt_fwd, prop_ei, device) -> float:
    """Stage 1: forward pre-training using ground-truth coordinates."""
    fwd_model.train()
    total = 0.0
    for batch in tqdm(loader, leave=False, disable=_TQDM_DISABLE):
        opt_fwd.zero_grad()
        di = batch["data_inv"].to(device)
        yt = batch["y_true"].to(device).squeeze(1)   # [B, 2]  ← FIX
        de = batch["delta_e_true"].to(device)         # [B, 36]
        gf = make_fwd_batch(di, prop_ei, device)
        loss = _focus_loss(fwd_model(gf, yt), de)
        loss.backward(); opt_fwd.step()
        total += loss.item() * di.num_graphs
    return total / len(loader.dataset)


def train_stage2_inv(inv_model, loader, opt_inv, device) -> float:
    """Stage 2: inverse warmup — MSE on coordinates only, no forward loss."""
    inv_model.train()
    crit = nn.MSELoss()
    total = 0.0
    for batch in tqdm(loader, leave=False, disable=_TQDM_DISABLE):
        opt_inv.zero_grad()
        di = batch["data_inv"].to(device)
        yt = batch["y_true"].to(device).squeeze(1)   # [B, 2]  ← FIX
        loss = crit(inv_model(di), yt)
        loss.backward(); opt_inv.step()
        total += loss.item() * di.num_graphs
    return total / len(loader.dataset)


def train_stage3_inv_phase(
    inv_model, fwd_model, loader, opt_inv, lam, prop_ei, device
) -> tuple[float, float]:
    """
    Stage 3 INV phase:
      inv_model.train(),  fwd_model.eval() (frozen teacher)
      Loss = L_inv + lam * L_fwd
      Only opt_inv is stepped.
    """
    inv_model.train()
    fwd_model.eval()
    crit    = nn.MSELoss()
    tot_inv = tot_fwd = 0.0

    for batch in tqdm(loader, leave=False, disable=_TQDM_DISABLE):
        opt_inv.zero_grad()
        di = batch["data_inv"].to(device)
        yt = batch["y_true"].to(device).squeeze(1)   # [B, 2]  ← FIX
        de = batch["delta_e_true"].to(device)         # [B, 36]

        pc    = inv_model(di)                         # [B, 2]
        l_inv = crit(pc, yt)

        gf    = make_fwd_batch(di, prop_ei, device)
        l_fwd = crit(fwd_model(gf, pc), de)

        (l_inv + lam * l_fwd).backward()
        opt_inv.step()

        n = di.num_graphs
        tot_inv += l_inv.item() * n
        tot_fwd += l_fwd.item() * n

    n = len(loader.dataset)
    return tot_inv / n, tot_fwd / n


def train_stage3_fwd_phase(
    inv_model, fwd_model, loader, opt_fwd, prop_ei, device
) -> float:
    """
    Stage 3 FWD-REFINE phase:
      fwd_model.train(),  inv_model.eval() (frozen)
      Pseudo-labels = inv_model predictions (no_grad)
      Loss = focus_loss on delta-E using predicted coordinates.
      Only opt_fwd is stepped.

    This forces the forward model to adapt to the quality of the CURRENT
    inverse predictions rather than remaining tied to ground-truth coordinates.
    """
    fwd_model.train()
    inv_model.eval()
    total = 0.0

    for batch in tqdm(loader, leave=False, disable=_TQDM_DISABLE):
        opt_fwd.zero_grad()
        di = batch["data_inv"].to(device)
        de = batch["delta_e_true"].to(device)          # [B, 36]

        # Predicted coords as pseudo-labels — detach so no grad through inv
        with torch.no_grad():
            pseudo_coords = inv_model(di)              # [B, 2], no grad

        gf   = make_fwd_batch(di, prop_ei, device)
        loss = _focus_loss(fwd_model(gf, pseudo_coords), de)
        loss.backward(); opt_fwd.step()
        total += loss.item() * di.num_graphs

    return total / len(loader.dataset)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_mae(model, loader, device) -> float:
    """
    Euclidean MAE over DAMAGED samples only, in normalised coords.
    Matches notebook Cell 47 (mean of batch-wise means, then average batches).
    Multiply by PLATE_MM for mm.
    """
    model.eval()
    total = 0.0; count = 0
    with torch.no_grad():
        for batch in loader:
            di   = batch["data_inv"].to(device)
            yt   = batch["y_true"].to(device).squeeze(1)   # [B, 2]  ← FIX
            mask = yt[:, 0] > 0                             # [B] bool
            if not mask.any():
                continue
            pred  = model(di)[mask]                         # [K, 2]
            true  = yt[mask]                                # [K, 2]
            total += torch.sqrt(((pred - true) ** 2).sum(dim=1)).mean().item()
            count += 1
    return (total / count) if count > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="WaveGraphNet alternating curriculum")

    # Data / split
    p.add_argument("--split",             default="A",   choices=["A", "B"])
    p.add_argument("--seed",              type=int,      default=42)
    p.add_argument("--batch_size",        type=int,      default=8)
    p.add_argument("--val_every",         type=int,      default=10)

    # Learning rates (separate per phase for fine-grained control)
    p.add_argument("--lr",                type=float,    default=1e-4,
                   help="Base LR for all optimizers")
    p.add_argument("--lr_fwd_refine",     type=float,    default=None,
                   help="LR for Stage-3 forward fine-tuning (default: 0.5*lr)")

    # Stage 1: forward pre-training
    p.add_argument("--stage1_epochs",     type=int,      default=500)

    # Stage 2: inverse warmup
    p.add_argument("--stage2_inv_epochs", type=int,      default=100)

    # Stage 3: alternating cycles
    p.add_argument("--stage3_cycles",     type=int,      default=8,
                   help="Number of alternating (inv + fwd-refine) cycles")
    p.add_argument("--inv_per_cycle",     type=int,      default=50,
                   help="Inverse-phase epochs per cycle")
    p.add_argument("--fwd_per_cycle",     type=int,      default=20,
                   help="Forward-refine epochs per cycle")
    p.add_argument("--warmup_cycles",     type=int,      default=2,
                   help="Lambda warmup in stage-3 cycles (lambda=0 for this many cycles)")
    p.add_argument("--max_lambda",        type=float,    default=100.0)

    # Model architecture
    p.add_argument("--inv_hidden_dim",         type=int, default=256)
    p.add_argument("--fwd_hidden_dim",         type=int, default=512)
    p.add_argument("--num_interaction_layers", type=int, default=8)
    p.add_argument("--gat_heads",              type=int, default=16)
    p.add_argument("--num_gnn_proc_layers",    type=int, default=4)

    args = p.parse_args()
    if args.lr_fwd_refine is None:
        args.lr_fwd_refine = args.lr * 0.5   # gentler LR for fine-tuning

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  {LABEL} | split={args.split} seed={args.seed}")
    print(f"  Stage1={args.stage1_epochs}ep | Stage2={args.stage2_inv_epochs}ep | "
          f"Stage3={args.stage3_cycles}×({args.inv_per_cycle}inv+{args.fwd_per_cycle}fwd)")
    print(f"  λ-warmup={args.warmup_cycles} cycles | λ-max={args.max_lambda} | "
          f"lr={args.lr} lr_fwd_refine={args.lr_fwd_refine}")
    print(f"{'='*65}\n", flush=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    with open("data/processed/ogw_data.pkl", "rb") as f:
        raw_data_map = pickle.load(f)

    train_ids, val_ids, test_ids = get_train_val_test_ids(
        args.split, list(raw_data_map.keys()), seed=args.seed)

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

    # ── Models ───────────────────────────────────────────────────────────────
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

    prop_ei    = stats["propagation_edge_index"]
    ckpt_root  = os.path.join("checkpoints", args.split, "trial")
    os.makedirs(ckpt_root, exist_ok=True)
    ckpt_best  = os.path.join(ckpt_root, f"best_seed{args.seed}.pt")
    ckpt_fwd   = os.path.join(ckpt_root, f"fwd_stage1_seed{args.seed}.pt")

    logger = EpochLogger(args.split, LABEL, args.seed)

    # Track best across all stages
    best_val_mae  = float("inf")
    best_test_mae = float("nan")
    best_test_mse = float("nan")
    global_epoch  = 0

    def checkpoint_if_best(vm, vm_mm, tm, tmm, tag=""):
        nonlocal best_val_mae, best_test_mae, best_test_mse
        if vm_mm < best_val_mae:
            best_val_mae  = vm_mm
            best_test_mae = tmm
            best_test_mse = tm
            save_checkpoint(ckpt_best, config=vars(args),
                            test_loss=tm, val_loss=vm_mm,
                            inv_model=inv_model.state_dict(),
                            fwd_model=fwd_model.state_dict())
            print(f"  ★ [{tag}] Ep {global_epoch:05d} | ValMAE={vm_mm:.1f}mm → "
                  f"TestMAE={tmm:.1f}mm TestMSE={tm:.5f} [saved]", flush=True)

    # ── Stage 1: forward pre-training ─────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  STAGE 1: Forward pre-training ({args.stage1_epochs} epochs)")
    print(f"{'─'*65}", flush=True)

    opt_fwd_s1 = optim.Adam(fwd_model.parameters(), lr=args.lr)
    sch_fwd_s1 = optim.lr_scheduler.ReduceLROnPlateau(opt_fwd_s1, factor=0.8, patience=20)
    best_s1_loss = float("inf")

    for ep in range(1, args.stage1_epochs + 1):
        global_epoch += 1
        fl = train_stage1_fwd(fwd_model, tr, opt_fwd_s1, prop_ei, device)
        sch_fwd_s1.step(fl)

        if ep % args.val_every == 0 or ep in (1, args.stage1_epochs):
            print(f"  [S1 Fwd] Ep {ep:04d} | FwdLoss={fl:.6f} "
                  f"| LR={opt_fwd_s1.param_groups[0]['lr']:.2e}", flush=True)

        if fl < best_s1_loss:
            best_s1_loss = fl
            torch.save(fwd_model.state_dict(), ckpt_fwd)

    print(f"  Stage 1 complete. Best fwd loss={best_s1_loss:.6f} → loading.", flush=True)
    fwd_model.load_state_dict(torch.load(ckpt_fwd, map_location=device))

    # ── Stage 2: inverse warmup ───────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  STAGE 2: Inverse warmup — no forward loss ({args.stage2_inv_epochs} epochs)")
    print(f"{'─'*65}", flush=True)

    opt_inv = optim.Adam(inv_model.parameters(), lr=args.lr)
    sch_inv = optim.lr_scheduler.ReduceLROnPlateau(opt_inv, factor=0.8, patience=20)

    for ep in range(1, args.stage2_inv_epochs + 1):
        global_epoch += 1
        il = train_stage2_inv(inv_model, tr, opt_inv, device)

        if ep % args.val_every == 0 or ep in (1, args.stage2_inv_epochs):
            vm  = euclidean_mae(inv_model, vl, device)
            vmm = vm * PLATE_MM
            sch_inv.step(vm)
            tm, tmm = evaluate_test(inv_model, te, device)
            logger.log(global_epoch, "S2_inv", il, vm, vmm, tm, tmm,
                       lambda_fwd=0.0, lr=opt_inv.param_groups[0]["lr"])
            checkpoint_if_best(vm, vmm, tm, tmm, tag="S2")

    # ── Stage 3: alternating coupled training ─────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  STAGE 3: Alternating — {args.stage3_cycles} cycles "
          f"({args.inv_per_cycle} inv + {args.fwd_per_cycle} fwd per cycle)")
    print(f"{'─'*65}", flush=True)

    # Separate optimizer for Stage-3 forward fine-tuning (lower LR)
    opt_fwd_s3 = optim.Adam(fwd_model.parameters(), lr=args.lr_fwd_refine)
    sch_fwd_s3 = optim.lr_scheduler.ReduceLROnPlateau(opt_fwd_s3, factor=0.8, patience=10)

    for cycle in range(args.stage3_cycles):
        lam = get_lambda_by_cycle(cycle, args.stage3_cycles,
                                   args.warmup_cycles, args.max_lambda)

        # ── INV phase ─────────────────────────────────────────────────────────
        print(f"\n  [Cycle {cycle+1}/{args.stage3_cycles}] λ={lam:.1f} — "
              f"INV phase ({args.inv_per_cycle} epochs)", flush=True)

        for ep in range(1, args.inv_per_cycle + 1):
            global_epoch += 1
            il, fl_consistency = train_stage3_inv_phase(
                inv_model, fwd_model, tr, opt_inv, lam, prop_ei, device)

            if ep % args.val_every == 0 or ep in (1, args.inv_per_cycle):
                vm  = euclidean_mae(inv_model, vl, device)
                vmm = vm * PLATE_MM
                sch_inv.step(vm)
                tm, tmm = evaluate_test(inv_model, te, device)
                logger.log(global_epoch, "S3_inv", il, vm, vmm, tm, tmm,
                           lambda_fwd=lam, lr=opt_inv.param_groups[0]["lr"])
                checkpoint_if_best(vm, vmm, tm, tmm, tag=f"S3-C{cycle+1}-inv")

        # ── FWD-REFINE phase ──────────────────────────────────────────────────
        print(f"  [Cycle {cycle+1}/{args.stage3_cycles}] — "
              f"FWD-REFINE phase ({args.fwd_per_cycle} epochs)", flush=True)

        for ep in range(1, args.fwd_per_cycle + 1):
            global_epoch += 1
            fl = train_stage3_fwd_phase(
                inv_model, fwd_model, tr, opt_fwd_s3, prop_ei, device)
            sch_fwd_s3.step(fl)

            if ep % args.val_every == 0 or ep in (1, args.fwd_per_cycle):
                print(f"    [Fwd-Refine] Ep {global_epoch:05d} | "
                      f"FwdLoss={fl:.6f} | LR={opt_fwd_s3.param_groups[0]['lr']:.2e}",
                      flush=True)

        # Evaluate after the full cycle
        vm  = euclidean_mae(inv_model, vl, device)
        vmm = vm * PLATE_MM
        tm, tmm = evaluate_test(inv_model, te, device)
        print(f"  [Cycle {cycle+1} END] ValMAE={vmm:.1f}mm | "
              f"TestMAE={tmm:.1f}mm | TestMSE={tm:.5f}", flush=True)
        checkpoint_if_best(vm, vmm, tm, tmm, tag=f"S3-C{cycle+1}-end")

    # ── Final reporting ────────────────────────────────────────────────────────
    logger.close()
    log_result(args.split, f"{LABEL} (seed={args.seed})", best_test_mse)
    log_mae_result(args.split, LABEL, args.seed, best_val_mae, best_test_mae)

    total_stage3 = args.stage3_cycles * (args.inv_per_cycle + args.fwd_per_cycle)
    print(f"\n{'='*65}")
    print(f"  {LABEL} | split={args.split} seed={args.seed}")
    print(f"  Total epochs: Stage1={args.stage1_epochs} + "
          f"Stage2={args.stage2_inv_epochs} + "
          f"Stage3={total_stage3} = {args.stage1_epochs + args.stage2_inv_epochs + total_stage3}")
    print(f"  Best val MAE : {best_val_mae:.1f} mm")
    print(f"  Best test MAE: {best_test_mae:.1f} mm")
    print(f"  Best test MSE: {best_test_mse:.5f}")
    print(f"{'='*65}", flush=True)


if __name__ == "__main__":
    main()