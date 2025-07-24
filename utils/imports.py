import sys
import os

# Show maps inline
from bokeh.resources import INLINE
import bokeh.io
bokeh.io.output_notebook(INLINE)

# Now you can import from config.py
from config import BASE_DIR, DATA_DIR
path_to_sqlite = str( DATA_DIR / "sqlite/belgium.sqlite" ) # Path where the sqlite database is stored
L_space_path= DATA_DIR / "pkl/belgium_routesCleaned.pkl"  # Path where the clean L-space graph was stored (cleaned routes)

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

attributes = load_gtfs(path_to_sqlite)
L_graph = load_graph(L_space_path)
subgraphs = load_all_subgraphs(DATA_DIR / "pkl")

from gtfs_railways.functions.v0 import (
    get_all_GTC as get_all_GTC_v0,
    P_space as P_space_v0)

from gtfs_railways.functions.v1 import get_all_GTC as get_all_GTC_v1, P_space as P_space_v1
from gtfs_railways.functions.v2 import get_all_GTC as get_all_GTC_v2, P_space as P_space_v2
from gtfs_railways.functions.v3 import get_all_GTC as get_all_GTC_v3, P_space as P_space_v3
from gtfs_railways.functions.v4 import get_all_GTC as get_all_GTC_v4, P_space as P_space_v4
from gtfs_railways.functions.v5 import get_all_GTC as get_all_GTC_v5, P_space as P_space_v5

