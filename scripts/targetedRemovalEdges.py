import sys
import time
import pickle
from pathlib import Path
from gtfs_railways.utils.config import DATA_DIR
from gtfs_railways.functions.core import (
    load_gtfs,
    load_graph,
    compute_time,
    make_sp_func,
    simulate_fixed_node_removal_efficiency,
    export_removal_results_to_csv
)
from gtfs_railways.functions.v4 import P_space as P_space_4, get_all_GTC as get_all_GTC_4

simulate_fixed_node_removal_efficiency = compute_time(simulate_fixed_node_removal_efficiency)

# ---------------- Config ----------------
checkpoint_interval = 100
seed = 42

# ---------- Choose dataset ----------
# Uncomment the dataset you want

# --- Belgium ---
attributes = load_gtfs(str(DATA_DIR / "sqlite/belgium.sqlite"))
L_graph = load_graph(DATA_DIR / "pkl/belgium_routesCleaned.pkl")
results_dir = DATA_DIR / "results/removal_runs_targeted_edge_BE"
log_path = results_dir / "removal_targeted_edges_BE.log"
print("Working on BE data")

# --- Netherlands ---
# attributes = load_gtfs(str(DATA_DIR / "sqlite/NL.sqlite"))
# L_graph = load_graph(DATA_DIR / "pkl/nl_merged.pkl")
# results_dir = DATA_DIR / "results/removal_runs_targeted_edge_NL"
# log_path = results_dir / "removal_targeted_edges_NL.log"
# print("Working on NL data")

results_dir.mkdir(parents=True, exist_ok=True)
sys.stdout = open(log_path, "a")
sys.stderr = sys.stdout

sp_func = make_sp_func(attributes, get_all_GTC_4, P_space_4)

# ---------- Resume from last checkpoint ----------
pkls = sorted(results_dir.glob("checkpoint_*.pkl"))
if pkls:
    latest_checkpoint = pkls[-1]
    with open(latest_checkpoint, "rb") as f:
        checkpoint_data = pickle.load(f)
    L_graph = checkpoint_data['graph']
    efficiencies = checkpoint_data['efficiencies']
    pct_remaining = checkpoint_data['percent_remaining']
    removed_edges = checkpoint_data['removed_edges']
    removal_times = checkpoint_data['removal_times']
    removed_count = len(removed_edges)
    print(f"Resuming from checkpoint {latest_checkpoint.name}: {removed_count} edges removed")
else:
    removed_count = 0
    efficiencies, pct_remaining, removed_edges, removal_times = [], [], [], []

total_edges = L_graph.number_of_edges() + removed_count
print(f"Total edges in full graph: {total_edges}")

# ---------- Run edge removal in batches ----------
try:
    edges_list = list(L_graph.edges())
    edges_to_remove = edges_list[removed_count:]  # continue from last checkpoint

    for batch_start in range(0, len(edges_to_remove), checkpoint_interval):
        batch_edges = edges_to_remove[batch_start:batch_start + checkpoint_interval]

        # Run your edge removal function for this batch
        orig_eff, batch_eff, batch_pct, batch_removed, batch_times = simulate_fixed_node_removal_efficiency(
            L_graph=L_graph,
            sp_func=sp_func,
            pct_to_remove=(removed_count + len(batch_edges)) / total_edges * 100,
            removal_type='edge',
            method='targeted',
            seed=seed,
            verbose=False
        )

        # Append batch results
        efficiencies.extend(batch_eff)
        pct_remaining.extend(batch_pct)
        removed_edges.extend(batch_removed)
        removal_times.extend(batch_times)
        removed_count += len(batch_edges)

        # Save checkpoint
        checkpoint_data = {
            'graph': L_graph,
            'efficiencies': efficiencies,
            'percent_remaining': pct_remaining,
            'removed_edges': removed_edges,
            'removal_times': removal_times
        }
        checkpoint_file = results_dir / f"checkpoint_{removed_count:06d}.pkl"
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)

        # Save partial CSV
        csv_file = results_dir / f"results_{removed_count:06d}.csv"
        export_removal_results_to_csv(
            output_path=csv_file,
            efficiencies=efficiencies,
            percent_remaining=pct_remaining,
            removed_nodes=removed_edges,
            removal_times=removal_times
        )
        print(f"Checkpoint and CSV saved: {removed_count} edges removed")

    # Final CSV
    final_csv = results_dir / f"targeted_edge_removal_seed{seed}_edges{total_edges}.csv"
    export_removal_results_to_csv(
        output_path=final_csv,
        efficiencies=efficiencies,
        percent_remaining=pct_remaining,
        removed_nodes=removed_edges,
        removal_times=removal_times
    )
    print("Simulation completed. Final results saved.")

finally:
    sys.stdout.close()
