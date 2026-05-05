"""
WaveGraphNet — GPU-parallel evaluation pipeline.

Runs all models across all seeds, 3 at a time (one per GPU).
Results written to results_mae.json (MAE localization only).

Usage:
  # Split B2
  python run_all.py --split B2 --gpus 0 1 2 --seeds 0 1 42

  # Split A
  python run_all.py --split A --gpus 0 1 2 --seeds 0 1 42

  # Quick smoke test (2 epochs)
  python run_all.py --split B2 --gpus 0 1 2 --seeds 0 --quick
"""
import subprocess, argparse, sys, json, os, time, statistics
from collections import defaultdict

# ── Model configurations ───────────────────────────────────────────────────────
# Each entry: (script, extra_args, label)
# WaveGraphNet 900-epoch budget:
#   Inverse Only: 900ep Stage 0 + 0ep fwd + 0ep Phase2   = 900 total
#   Coupled:      150ep Stage 0 + 150ep fwd + 600ep Phase2 = 900 total

MODELS = [
    ("main_cnn.py",
     ["--epochs", "900"],
     "1D CNN"),

    ("main_lstm.py",
     ["--lstm_hidden_dim", "256", "--num_lstm_layers", "3", "--dropout", "0.3",
      "--epochs", "900"],
     "LSTM"),

    ("main_gnn_baselines.py",
     ["--model", "simple_mlp", "--hidden_dim", "256", "--num_gnn_layers", "4",
      "--epochs", "900"],
     "GNN (simple_mlp)"),

    ("main_gnn_baselines.py",
     ["--model", "attention", "--hidden_dim", "256",
      "--num_gnn_layers", "4", "--gat_heads", "16",
      "--epochs", "900"],
     "GNN (attention)"),

    # Inverse Only — dedicated script, 900 epochs, no forward model
    ("main_wavegraphnet_inv.py",
     ["--inv_hidden_dim", "256", "--num_gnn_proc_layers", "4", "--gat_heads", "16",
      "--fwd_hidden_dim", "128", "--num_interaction_layers", "3",
      "--epochs", "900",
      "--lr", "1e-4"],
     "WaveGraphNet (Inverse Only)"),

    # Coupled — 150+150+600 = 900 epochs, physics correction active
    ("main_wavegraphnet.py",
     ["--mode", "coupled",
      "--inv_hidden_dim", "256", "--num_gnn_proc_layers", "4", "--gat_heads", "16",
      "--fwd_hidden_dim", "128", "--num_interaction_layers", "3",
      "--inv_pretrain_epochs", "150",
      "--fwd_pretrain_epochs", "150",
      "--epochs", "600",
      "--warmup", "40", "--max_lambda", "3",
      "--mu", "1.0", "--alpha", "0.1", "--ckpt_alpha", "1.0",
      "--lr", "1e-4", "--lr_phase2", "1e-5"],
     "WaveGraphNet (Coupled)"),
]

QUICK_EPOCHS = "2"
GLOBAL_EPOCHS = "900"  # same budget for all models



def build_cmd(script, extra_args, split, seed, gpu, quick):
    """Build the full subprocess command for one run."""
    cmd = [
        sys.executable, script,
        "--split", split,
        "--seed",  str(seed),
    ] + extra_args

    if quick:
        # Override all epoch-related flags to 2 for smoke testing
        epoch_flags = [
            "--epochs", "--inv_pretrain_epochs",
            "--fwd_pretrain_epochs", "--fwd_epochs",
        ]
        seen = set()
        new_cmd = []
        skip_next = False
        for i, tok in enumerate(cmd):
            if skip_next:
                skip_next = False
                new_cmd.append(QUICK_EPOCHS)
                continue
            if tok in epoch_flags and tok not in seen:
                seen.add(tok)
                new_cmd.append(tok)
                skip_next = True
            else:
                new_cmd.append(tok)
        cmd = new_cmd

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return cmd, env


def log_path(script, split, seed):
    label = script.replace(".py", "").replace("main_", "")
    return f"logs/run_{label}_{split}_seed{seed}.log"


def run_batch(batch, quick):
    """Launch a batch of (cmd, env, logfile) concurrently, wait for all."""
    procs = []
    for cmd, env, logfile in batch:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        print(f"  → {' '.join(cmd[:6])} ... seed={cmd[cmd.index('--seed')+1]} "
              f"[GPU {env['CUDA_VISIBLE_DEVICES']}]")
        f = open(logfile, "w")
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=f)
        procs.append((p, f, logfile))
        time.sleep(2)   # stagger launches to avoid race on data loading

    print(f"  Waiting for {len(procs)} processes …")
    failed = []
    for p, f, logfile in procs:
        rc = p.wait()
        f.close()
        if rc != 0:
            failed.append((logfile, rc))
            print(f"  [ERROR] {logfile} exited with code {rc}")
        else:
            print(f"  [DONE]  {logfile}")
    return failed


def print_mae_leaderboard(split, seeds, results_path="results_mae.json"):
    if not os.path.exists(results_path):
        print("results_mae.json not found.")
        return
    with open(results_path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Could not read results_mae.json.")
            return

    split_data = data.get(split, {})
    if not split_data:
        print(f"No MAE results for Split {split}.")
        return

    rows = []
    for model_name, entries in split_data.items():
        maes = [e["test_mae_mm"] for e in entries
                if isinstance(e, dict) and "test_mae_mm" in e]
        if maes:
            rows.append((model_name,
                         statistics.mean(maes),
                         statistics.stdev(maes) if len(maes) > 1 else 0.0,
                         len(maes)))
    rows.sort(key=lambda r: r[1])

    print(f"\n{'='*65}")
    print(f"  MAE RESULTS  |  SPLIT {split}  |  seeds={seeds}")
    print(f"{'='*65}")
    print(f"{'Model':<38} | {'Mean MAE':>9} | {'Std':>7} | n")
    print("-" * 65)
    for name, mean, std, n in rows:
        print(f"{name:<38} | {mean:>7.1f}mm | {std:>5.1f}mm | {n}")
    print(f"{'='*65}")

    log = f"mae_summary_split{split}.log"
    with open(log, "w") as f:
        f.write(f"SPLIT {split} | seeds={seeds}\n")
        f.write(f"{'Model':<38} | {'Mean MAE':>9} | {'Std':>7}\n")
        f.write("-"*65+"\n")
        for name, mean, std, n in rows:
            f.write(f"{name:<38} | {mean:>7.1f}mm | {std:>5.1f}mm\n")
    print(f"[Saved] → {log}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split",  default="B2", choices=["A", "B", "B2"])
    p.add_argument("--gpus",   type=int, nargs="+", default=[0, 1, 2],
                   help="GPU IDs to use. Max concurrent jobs = len(gpus).")
    p.add_argument("--seeds",  type=int, nargs="+", default=[0, 1, 42])
    p.add_argument("--quick",  action="store_true",
                   help="Smoke test: force 2 epochs per model")
    p.add_argument("--models", type=int, nargs="+", default=None,
                   help="Run only specific model indices (0-5). Default: all.")
    args = p.parse_args()

    n_gpus = len(args.gpus)
    models = [MODELS[i] for i in args.models] if args.models else MODELS

    # Build full list of (script, extra_args, label, split, seed, gpu, logfile)
    jobs = []
    gpu_idx = 0
    for seed in args.seeds:
        for script, extra_args, label in models:
            gpu = args.gpus[gpu_idx % n_gpus]
            logfile = log_path(f"{script}_{label.replace(' ','_')}", args.split, seed)
            cmd, env = build_cmd(script, extra_args, args.split, seed, gpu, args.quick)
            jobs.append((cmd, env, logfile))
            gpu_idx += 1

    total = len(jobs)
    print(f"\n{'='*65}")
    print(f"  WaveGraphNet Pipeline | Split={args.split} | "
          f"Seeds={args.seeds} | GPUs={args.gpus}")
    print(f"  {total} jobs, {n_gpus} at a time")
    print(f"{'='*65}\n")

    # Clear previous results for this split
    for rfile in ("results.json", "results_mae.json"):
        if os.path.exists(rfile):
            with open(rfile) as f:
                try:
                    d = json.load(f)
                except json.JSONDecodeError:
                    d = {}
            d.pop(args.split, None)
            with open(rfile, "w") as f:
                json.dump(d, f, indent=4)

    # Run in batches of n_gpus
    all_failed = []
    for i in range(0, total, n_gpus):
        batch = jobs[i:i + n_gpus]
        batch_num = i // n_gpus + 1
        total_batches = (total + n_gpus - 1) // n_gpus
        print(f"\n── Batch {batch_num}/{total_batches} "
              f"(jobs {i+1}–{min(i+n_gpus, total)}/{total}) ──")
        failed = run_batch(batch, args.quick)
        all_failed.extend(failed)

    # Final leaderboard
    print_mae_leaderboard(args.split, args.seeds)

    if all_failed:
        print(f"\n[WARNING] {len(all_failed)} job(s) failed:")
        for logfile, rc in all_failed:
            print(f"  {logfile} (exit code {rc})")
        sys.exit(1)
    else:
        print(f"\n[DONE] All {total} jobs completed successfully.")


if __name__ == "__main__":
    main()