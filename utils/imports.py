import sys
import os

# Show maps inline
from bokeh.resources import INLINE
import bokeh.io
bokeh.io.output_notebook(INLINE)

# Now you can import from config.py
from config import BASE_DIR, DATA_DIR, PATH_TO_SQLITE, L_SPACE_PATH, P_SPACE_PATH

from gtfs_railways.functions.core import load_gtfs
from gtfs_railways.functions.core import load_graph
from gtfs_railways.functions.core import generate_subgraph_batches
from gtfs_railways.functions.core import save_subgraphs_by_size
from gtfs_railways.functions.core import load_all_subgraphs
from gtfs_railways.functions.core import efficiency_graph
from gtfs_railways.functions.core import make_sp_func
from gtfs_railways.functions.core import simulate_fixed_node_removal_efficiency
from gtfs_railways.functions.core import random_node_removal
from gtfs_railways.functions.core import targeted_node_removal
from gtfs_railways.functions.core import betweenness_node_removal
from gtfs_railways.functions.core import run_removal_simulations
from gtfs_railways.functions.core import compute_time
from gtfs_railways.functions.core import get_runtime
from gtfs_railways.functions.core import generate_subgraph_batches
from gtfs_railways.functions.core import compute_graph_features
from gtfs_railways.functions.core import get_efficiency_curves
from gtfs_railways.functions.core import get_random_removal_nodes
from gtfs_railways.functions.core import average_waiting_time_per_line_per_direction
from gtfs_railways.functions.core import average_speed_network
from gtfs_railways.functions.core import get_events
from gtfs_railways.functions.core import save_graph
from gtfs_railways.functions.core import save_gtc_to_pkl
from gtfs_railways.functions.core import load_gtc_from_pkl
from gtfs_railways.functions.core import betweenness_fit_revised
from gtfs_railways.functions.core import meshedness
from gtfs_railways.functions.core import plot_graph_highlight_node
from gtfs_railways.functions.core import plot_top_hubs
from gtfs_railways.functions.core import mode_to_string
from gtfs_railways.functions.core import generate_graph
from gtfs_railways.functions.core import merge_stops_with_same_name
from gtfs_railways.functions.core import check_islands
from gtfs_railways.functions.core import merge_recommender
from gtfs_railways.functions.core import manual_merge
from gtfs_railways.functions.core import sanity_check
from gtfs_railways.functions.core import process_route_data
from gtfs_railways.functions.core import node_degrees_table
from gtfs_railways.functions.core import edge_merger
from gtfs_railways.functions.core import load_removal_results_df
from gtfs_railways.functions.core import export_removal_results_to_csv

from notebooks.functions.plot import plot_efficiency_results
from notebooks.functions.plot import plot_graph
from notebooks.functions.plot import plot_efficiency_results_from_batch
from notebooks.functions.plot import compute_avg_runtime_by_num_nodes
from notebooks.functions.plot import plot_removal_time_vs_steps
from notebooks.functions.plot import plot_efficiency_decay
from notebooks.functions.plot import remove_node_edges_and_plot
from notebooks.functions.plot import plot_runtime_comparison
from notebooks.functions.plot import num_route_dir_pairs_with_density
from notebooks.functions.plot import sort_subgraphs_dict_by_route_dir_pairs
from notebooks.functions.plot import plot_runtime_bars
from notebooks.functions.plot import plot_runtime_vs_density_scatter
from notebooks.functions.plot import plot_efficiency_results_multi
from notebooks.functions.plot import analyze_runtime_improvement
from notebooks.functions.plot import plot_efficiency_from_loaded_df
from notebooks.functions.plot import plot_multiple_efficiency_runs
from notebooks.functions.plot import plot_average_efficiency_with_area
from notebooks.functions.plot import plot_efficiency_with_node_labels
from notebooks.functions.plot import plot_efficiency_with_node_labels_from_df
from notebooks.functions.plot import plot_efficiency_comparison_multi
from notebooks.functions.plot import plot_efficiency_comparison_single

from gtfs_railways.functions.v0 import (
    get_all_GTC as get_all_GTC_v0,
    P_space as P_space_v0)

from gtfs_railways.functions.v1 import get_all_GTC as get_all_GTC_v1, P_space as P_space_v1
from gtfs_railways.functions.v2 import get_all_GTC as get_all_GTC_v2, P_space as P_space_v2
from gtfs_railways.functions.v3 import get_all_GTC as get_all_GTC_v3, P_space as P_space_v3
from gtfs_railways.functions.v4 import get_all_GTC as get_all_GTC_v4, P_space as P_space_v4
from gtfs_railways.functions.v5 import get_all_GTC as get_all_GTC_v5, P_space as P_space_v5

