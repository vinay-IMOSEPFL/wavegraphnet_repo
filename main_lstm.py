"""
Bidirectional LSTM baseline — spectral features.

FIX vs previous version:
  • Uses build_all_stats() → normalized_data_map, matching all other models.
  • amp_means/amp_stds from normalized differential data (not raw signals).
"""
import argparse, sys, random, pickle
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, scipy.fft
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.splits import get_train_val_test_ids
from utils.data_loader import parse_damage_label, DAMAGE_LABELS
from utils.precompute import build_all_stats, LOOKBACK_POINTS, N_ATTENTION_FREQS, FIXED_FFT_BIN_INDICES
from utils.checkpointer import save_checkpoint, checkpoint_path
from models.lstm import LSTM_baseline
from utils.logger import log_result, log_mae_result, EpochLogger, evaluate_test

_TQDM_DISABLE = not sys.stdout.isatty()
PLATE_MM      = 500.0


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


class LstmDataset(Dataset):
    """Each sample → tensor [66, 256, 2] (pairs, freqs, [norm_amp, phase])."""
    def __init__(self, data_map, sample_id_list, bins, amp_means, amp_stds, lookback):
        self.data_map   = data_map
        self.sample_ids = sample_id_list
        self.lookback   = lookback
        self.bins       = bins
        self.amp_means  = torch.tensor(amp_means, dtype=torch.float32)  # [66, 256]
        self.amp_stds   = torch.tensor(amp_stds,  dtype=torch.float32)
        self.nf         = len(bins)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        sig = self.data_map[sid]     # normalized differential [T, 66]

        ff     = scipy.fft.rfft(sig[:self.lookback, :], n=self.lookback, axis=0)
        amps   = torch.from_numpy(np.abs(ff[self.bins, :]).astype(np.float32)).T    # [66, 256]
        phases = torch.from_numpy(np.angle(ff[self.bins, :]).astype(np.float32)).T  # [66, 256]
        norm_amps = (amps - self.amp_means) / self.amp_stds
        x = torch.stack([norm_amps, phases], dim=-1)   # [66, 256, 2]

        dmg = parse_damage_label(sid)
        xd, yd = -0.001, -0.001
        if dmg != "undamaged" and dmg in DAMAGE_LABELS:
            xd, yd = DAMAGE_LABELS[dmg]
        return x, torch.tensor([xd, yd], dtype=torch.float)


def _euclidean_mae(model, loader, device) -> float:
    model.eval(); total = 0.0; count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            mask = y[:, 0] > 0
            if not mask.any(): continue
            err = torch.sqrt(((model(x)[mask] - y[mask]) ** 2).sum(dim=1)).mean().item()
            total += err; count += 1
    return total / count if count > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split",           default="A", choices=["A", "B"])
    p.add_argument("--epochs",          type=int,   default=500)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--lstm_hidden_dim", type=int,   default=256)
    p.add_argument("--num_lstm_layers", type=int,   default=3)
    p.add_argument("--dropout",         type=float, default=0.3)
    p.add_argument("--val_every",       type=int,   default=10)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/processed/ogw_data.pkl", "rb") as f:
        raw = pickle.load(f)

    train_ids, val_ids, test_ids = get_train_val_test_ids(
        args.split, list(raw.keys()), seed=args.seed)

    print(f"\n{'='*65}\n  LSTM | split={args.split} seed={args.seed}\n{'='*65}",
          flush=True)

    # FIX: use build_all_stats for normalized differential data
    stats     = build_all_stats(raw, train_ids)
    norm_data = stats["normalized_data_map"]
    num_pairs = list(norm_data.values())[0].shape[1]

    ds_kw = dict(
        data_map  = norm_data,
        bins      = FIXED_FFT_BIN_INDICES,
        amp_means = stats["amp_means"],
        amp_stds  = stats["amp_stds"],
        lookback  = LOOKBACK_POINTS,
    )
    tr = DataLoader(LstmDataset(sample_id_list=train_ids, **ds_kw),
                    batch_size=args.batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(LstmDataset(sample_id_list=val_ids,   **ds_kw),
                    batch_size=args.batch_size, shuffle=False, num_workers=0)
    te = DataLoader(LstmDataset(sample_id_list=test_ids,  **ds_kw),
                    batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LSTM_baseline(
        num_freqs            = N_ATTENTION_FREQS,
        feature_dim_per_freq = 2,
        num_sensor_pairs     = num_pairs,
        lstm_hidden_dim      = args.lstm_hidden_dim,
        num_lstm_layers      = args.num_lstm_layers,
        decoder_hidden_dim   = args.lstm_hidden_dim,
        output_dim           = 2,
        dropout_rate         = args.dropout,
    ).to(device)

    opt    = optim.Adam(model.parameters(), lr=args.lr)
    sch    = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.8, patience=20)
    crit   = nn.MSELoss()
    ckpt   = checkpoint_path(args.split, "LSTM", args.seed)
    logger = EpochLogger(args.split, "LSTM", args.seed)

    best_vm = float("inf"); best_tm = float("inf"); best_tmm = float("nan")

    for epoch in range(1, args.epochs + 1):
        model.train(); tl = 0.0
        for x, y in tqdm(tr, leave=False, disable=_TQDM_DISABLE):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y); loss.backward(); opt.step()
            tl += loss.item() * x.size(0)
        tl /= len(tr.dataset)

        if epoch % args.val_every == 0 or epoch in (1, args.epochs):
            val_mae_norm = _euclidean_mae(model, vl, device)
            val_mae_mm   = val_mae_norm * PLATE_MM
            sch.step(val_mae_norm)
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
    log_result(args.split, f"LSTM (seed={args.seed})", best_tm)
    log_mae_result(args.split, "LSTM", args.seed, best_vm, best_tmm)
    print(f"\n[DONE] LSTM | best_val_mae={best_vm:.1f}mm | "
          f"test_mae={best_tmm:.1f}mm | test_mse={best_tm:.5f}", flush=True)


if __name__ == "__main__":
    main()