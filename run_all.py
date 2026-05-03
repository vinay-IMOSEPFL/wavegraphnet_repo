import subprocess
import argparse
import sys
import json
import os
import statistics
from collections import defaultdict


# ---------------------------------------------------------------------------
# Paper-exact configuration (from paper Table)
# ---------------------------------------------------------------------------
PAPER_GLOBAL_ARGS = [
    "--lr",         "1e-4",
    "--batch_size", "8",
    "--epochs",     "500",
    # FFT bins and lookback are now determined by physical constants
    # inside utils/precompute.py (69.4-128 kHz, 256 bins, 13108-pt FFT).
]

# (script, script-specific args, display label)
PAPER_SCRIPTS = [
    ("main_cnn.py",
     [],
     "1D CNN"),

    ("main_lstm.py",
     ["--lstm_hidden_dim", "256", "--num_lstm_layers", "3", "--dropout", "0.3"],
     "LSTM"),

    ("main_gnn_baselines.py",
     ["--model", "simple_mlp", "--hidden_dim", "256", "--num_gnn_layers", "4"],
     "GNN (simple_mlp)"),

    ("main_gnn_baselines.py",
     ["--model", "attention", "--hidden_dim", "256",
      "--num_gnn_layers", "4", "--gat_heads", "16"],
     "GNN (attention)"),

    ("main_wavegraphnet.py",
     ["--mode", "inverse_only",
      "--inv_hidden_dim", "256", "--fwd_hidden_dim", "512",
      "--num_interaction_layers", "8", "--gat_heads", "16",
      "--num_gnn_proc_layers", "4",
      "--fwd_epochs", "500", "--warmup", "100", "--max_lambda", "100"],
     "WaveGraphNet (Inverse Only)"),

    ("main_wavegraphnet.py",
     ["--mode", "coupled",
      "--inv_hidden_dim", "256", "--fwd_hidden_dim", "512",
      "--num_interaction_layers", "8", "--gat_heads", "16",
      "--num_gnn_proc_layers", "4",
      "--fwd_epochs", "500", "--warmup", "100", "--max_lambda", "100"],
     "WaveGraphNet (Coupled)")
]

# Default scripts (when --paper is NOT set)
DEFAULT_SCRIPTS = [
    ("main_cnn.py",              [],                         "1D CNN"),
    ("main_lstm.py",             [],                         "LSTM"),
    ("main_gnn_baselines.py",    [],                         "GNN (simple_mlp)"),
    ("main_wavegraphnet.py",     ["--mode", "inverse_only"], "WaveGraphNet (Inverse Only)"),
    ("main_wavegraphnet.py",     ["--mode", "coupled"],      "WaveGraphNet (Coupled)"),
]
# ---------------------------------------------------------------------------


def _set_or_replace(arg_list, flag, value):
    """Insert or overwrite a --flag value pair in an arg list in-place."""
    if flag in arg_list:
        arg_list[arg_list.index(flag) + 1] = value
    else:
        arg_list.extend([flag, value])


def run_script(script_name, split, seed, quick_mode=False, extra_args=None):
    """Build and execute one child training command."""
    cmd = [sys.executable, script_name, "--split", split, "--seed", str(seed)]
    if extra_args:
        cmd.extend(extra_args)
    if quick_mode:
        _set_or_replace(cmd, "--epochs", "2")
        # Also reduce WaveGraphNet-specific long-running flags
        for flag in ("--fwd_epochs", "--stage1_epochs", "--stage2_inv_epochs",
                     "--inv_per_cycle", "--fwd_per_cycle"):
            _set_or_replace(cmd, flag, "2")

    print(f"\n{'=' * 65}")
    print(f"Executing: {' '.join(cmd)}")
    print(f"{'=' * 65}")
    subprocess.run(cmd, check=True)


def print_leaderboard(split, seeds, results_path="results.json"):
    """Print mean ± std across seeds from results.json."""
    if not os.path.exists(results_path):
        print("results.json not found.")
        return

    with open(results_path, "r") as f:
        try:
            all_results = json.load(f)
        except json.JSONDecodeError:
            print("Could not read results.json.")
            return

    split_data = all_results.get(split, {})
    if not split_data:
        print(f"No results logged for Split {split}.")
        return

    # Group entries by base model name (strip " (seed=N)" suffix).
    grouped = defaultdict(list)
    for key, loss in split_data.items():
        base = key.rsplit(" (seed=", 1)[0]
        grouped[base].append(loss)

    print(f"\n\n{'=' * 62}")
    print(f"  FINAL BASELINE RESULTS  |  SPLIT {split}  |  seeds={seeds}")
    print(f"{'=' * 62}")
    print(f"{'Model Name':<36} | {'Mean MSE':>10} | {'Std':>8}")
    print("-" * 62)
    for model, losses in sorted(grouped.items(), key=lambda x: statistics.mean(x[1])):
        mean = statistics.mean(losses)
        std  = statistics.stdev(losses) if len(losses) > 1 else 0.0
        print(f"{model:<36} | {mean:>10.6f} | {std:>8.6f}")


def print_mae_leaderboard(split, seeds, results_path="results_mae.json"):
    """Print mean ± std MAE (mm) across seeds from results_mae.json."""
    import statistics as st
    if not os.path.exists(results_path):
        print("results_mae.json not found — no MAE summary available.")
        return

    with open(results_path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Could not read results_mae.json.")
            return

    split_data = data.get(split, {})
    if not split_data:
        print(f"No MAE results logged for Split {split}.")
        return

    rows = []
    for model_name, entries in split_data.items():
        maes = [e["test_mae_mm"] for e in entries
                if isinstance(e, dict) and "test_mae_mm" in e]
        if maes:
            rows.append((model_name, st.mean(maes),
                         st.stdev(maes) if len(maes) > 1 else 0.0,
                         len(maes)))

    rows.sort(key=lambda r: r[1])

    print(f"\n\n{'='*65}")
    print(f"  FINAL MAE RESULTS  |  SPLIT {split}  |  seeds={seeds}")
    print(f"{'='*65}")
    print(f"{'Model Name':<38} | {'Mean MAE':>10} | {'Std':>8} | n")
    print("-" * 65)
    for name, mean, std, n in rows:
        print(f"{name:<38} | {mean:>8.1f}mm | {std:>6.1f}mm | {n}")
    print(f"{'='*65}")

    # Also write to a log file for nohup runs
    log_path = f"mae_summary_split{split}.log"
    with open(log_path, "w") as f:
        f.write(f"SPLIT {split} | seeds={seeds}\n")
        f.write(f"{'Model':<38} | {'Mean MAE':>10} | {'Std':>8}\n")
        f.write("-" * 65 + "\n")
        for name, mean, std, n in rows:
            f.write(f"{name:<38} | {mean:>8.1f}mm | {std:>6.1f}mm\n")
    print(f"[Saved] MAE summary → {log_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run the full WaveGraphNet evaluation pipeline.")
    parser.add_argument("--split", type=str, default="B", choices=["A", "B"])
    parser.add_argument("--skip_data", action="store_true",
                        help="Skip the data preparation phase")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: force 2 epochs per model")

    # ---- paper preset ----
    parser.add_argument("--paper", action="store_true",
                        help="Use exact paper hyperparameters: "
                             "lr=1e-4, batch=8, epochs=500, "
                             "per-model hidden dims, seeds {0,1,42}")

    # ---- manual overrides (work both with and without --paper) ----
    parser.add_argument("--epochs",       type=int,   default=None,
                        help="Override number of training epochs")
    parser.add_argument("--batch_size",   type=int,   default=None,
                        help="Override batch size")
    parser.add_argument("--lr",           type=float, default=None,
                        help="Override learning rate")

    # ---- seed control ----
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Random seeds to average over. "
                             "Default: [42] normally, [0,1,42] with --paper")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Resolve effective configuration                                       #
    # ------------------------------------------------------------------ #
    if args.paper:
        scripts   = PAPER_SCRIPTS
        base_args = list(PAPER_GLOBAL_ARGS)
        seeds     = args.seeds or [0, 1, 42]
    else:
        scripts   = DEFAULT_SCRIPTS
        base_args = []
        seeds     = args.seeds or [42]

    # Apply any manual overrides on top.
    for flag, val in [
        ("--epochs",       args.epochs),
        ("--batch_size",   args.batch_size),
        ("--lr",           args.lr),
    ]:
        if val is not None:
            _set_or_replace(base_args, flag, str(val))

    # ------------------------------------------------------------------ #
    # Clear previous results for this split                                #
    # ------------------------------------------------------------------ #
    if not args.quick:
        for rfile in ("results.json", "results_mae.json"):
            if os.path.exists(rfile):
                with open(rfile, "r") as f:
                    try:
                        all_results = json.load(f)
                    except json.JSONDecodeError:
                        all_results = {}
                all_results.pop(args.split, None)
                with open(rfile, "w") as f:
                    json.dump(all_results, f, indent=4)

    # ------------------------------------------------------------------ #
    # Data preparation                                                      #
    # ------------------------------------------------------------------ #
    if not args.skip_data:
        print("Preparing Data...")
        try:
            subprocess.run([sys.executable, "data/prepare_data.py"], check=True)
        except subprocess.CalledProcessError:
            print("Data preparation failed. Exiting.")
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Training loop — seeds × scripts                                      #
    # ------------------------------------------------------------------ #
    for seed in seeds:
        print(f"\n{'#' * 65}")
        print(f"#  SEED = {seed}")
        print(f"{'#' * 65}")

        for script, script_extra, _label in scripts:
            # Script-specific args come first so global base_args can override.
            combined = script_extra + base_args
            try:
                run_script(
                    script,
                    args.split,
                    seed=seed,
                    quick_mode=args.quick,
                    extra_args=combined or None,
                )
            except subprocess.CalledProcessError:
                print(f"\n[ERROR] {script} with args {combined} failed. "
                      "Exiting pipeline.")
                sys.exit(1)

    # ------------------------------------------------------------------ #
    # Leaderboard                                                           #
    # ------------------------------------------------------------------ #
    print_leaderboard(args.split, seeds)
    print_mae_leaderboard(args.split, seeds)


if __name__ == "__main__":
    main()