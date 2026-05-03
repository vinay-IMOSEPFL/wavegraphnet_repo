import argparse, sys, random, pickle
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from utils.splits import get_train_val_test_ids
from utils.data_loader import StandardGraphDataset
from utils.precompute import build_all_stats, N_ATTENTION_FREQS
from utils.checkpointer import save_checkpoint, checkpoint_path
from models.gnn_baselines import FlexibleGNN
from utils.logger import log_result, log_mae_result, EpochLogger, evaluate_val, evaluate_test

PLATE_MM = 500.0
_TQDM_DISABLE = not sys.stdout.isatty()

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def _euclidean_mae(model, loader, device):
    model.eval(); total = 0.0; count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            yt = batch.y.squeeze(1); mask = yt[:, 0] > 0
            if not mask.any(): continue
            err = torch.sqrt(((model(batch)[mask] - yt[mask])**2).sum(dim=1)).mean().item()
            total += err; count += 1
    return total/count if count>0 else float("nan")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split",          default="A", choices=["A","B"])
    p.add_argument("--model",          default="simple_mlp", choices=["simple_mlp","attention"])
    p.add_argument("--epochs",         type=int,  default=500)
    p.add_argument("--batch_size",     type=int,  default=8)
    p.add_argument("--lr",             type=float,default=1e-4)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--hidden_dim",     type=int,  default=256)
    p.add_argument("--num_gnn_layers", type=int,  default=4)
    p.add_argument("--gat_heads",      type=int,  default=16)
    p.add_argument("--val_every",      type=int,  default=10)
    args = p.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/processed/ogw_data.pkl","rb") as f: raw = pickle.load(f)
    train_ids, val_ids, test_ids = get_train_val_test_ids(args.split, list(raw.keys()), seed=args.seed)
    label = f"GNN ({args.model})"
    print(f"\n{'='*65}\n  {label} | split={args.split} seed={args.seed}\n{'='*65}", flush=True)

    stats = build_all_stats(raw, train_ids)
    norm  = stats["normalized_data_map"]
    ef    = 3 + N_ATTENTION_FREQS * 2
    ds_kw = dict(static_edge_index=stats["k12_edge_index"],
                 edge_feature_col_idxs=stats["inv_edge_feature_col_idxs"],
                 fixed_fft_bin_indices=stats["fixed_fft_bin_indices"],
                 amp_means=stats["amp_means"], amp_stds=stats["amp_stds"],
                 lookback_fft=stats["lookback_fft"])
    tr = DataLoader(StandardGraphDataset(norm, train_ids, **ds_kw), batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(StandardGraphDataset(norm, val_ids,   **ds_kw), batch_size=args.batch_size, shuffle=False)
    te = DataLoader(StandardGraphDataset(norm, test_ids,  **ds_kw), batch_size=args.batch_size, shuffle=False)

    model = FlexibleGNN(encoder_type=args.model, processor_type="mlp",
                        raw_node_feat_dim=2, raw_edge_feat_dim=ef,
                        num_attention_freqs=N_ATTENTION_FREQS,
                        hidden_dim=args.hidden_dim, num_gnn_proc_layers=args.num_gnn_layers,
                        gat_attention_heads=args.gat_heads,
                        decoder_mlp_hidden_dim=args.hidden_dim, final_output_dim=2,
                        decoder_pooling_type="max",   # matches notebook
                        num_decoder_mlp_layers=3, decoder_dropout_rate=0.2).to(device)
    opt  = optim.Adam(model.parameters(), lr=args.lr)
    sch  = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.8, patience=20)
    crit = nn.MSELoss()
    ckpt = checkpoint_path(args.split, label, args.seed)
    logger = EpochLogger(args.split, label, args.seed)
    best_vm = float("inf"); best_tm = float("inf"); best_tmm = float("nan")

    for epoch in range(1, args.epochs + 1):
        model.train(); tl = 0.0
        for batch in tqdm(tr, leave=False, disable=_TQDM_DISABLE):
            batch = batch.to(device); opt.zero_grad()
            loss = crit(model(batch), batch.y.squeeze(1))
            loss.backward(); opt.step()
            tl += loss.item() * batch.num_graphs
        tl /= len(tr.dataset)

        if epoch % args.val_every == 0 or epoch in (1, args.epochs):
            val_mae_norm = _euclidean_mae(model, vl, device)
            val_mae_mm   = val_mae_norm * PLATE_MM
            sch.step(val_mae_norm)   # scheduler on val MAE, matches notebook
            tm, tmm = evaluate_test(model, te, device)
            logger.log(epoch, "train", tl, val_mae_norm, val_mae_mm, tm, tmm,
                       lr=opt.param_groups[0]["lr"])
            if val_mae_mm < best_vm:
                best_vm = val_mae_mm; best_tm = tm; best_tmm = tmm
                save_checkpoint(ckpt, config=vars(args), test_loss=tm,
                                val_loss=val_mae_mm, model=model.state_dict())
                print(f"  ★ Ep {epoch:04d} | ValMAE={val_mae_mm:.1f}mm → "
                      f"TestMAE={tmm:.1f}mm TestMSE={tm:.5f} [saved]", flush=True)

    logger.close()
    log_result(args.split, f"{label} (seed={args.seed})", best_tm)
    log_mae_result(args.split, label, args.seed, best_vm, best_tmm)
    print(f"\n[DONE] {label} | best_val_mae={best_vm:.1f}mm | "
          f"test_mae={best_tmm:.1f}mm | test_mse={best_tm:.5f}", flush=True)

if __name__ == "__main__": main()