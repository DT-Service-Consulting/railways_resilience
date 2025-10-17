import matplotlib.pyplot as plt # type: ignore
import matplotlib.cm as cm # type: ignore
from matplotlib.lines import Line2D # type: ignore
import numpy as np 
import pandas as pd # type: ignore
from scipy import integrate # type: ignore
from pathlib import Path
import re

from bokeh.plotting import figure, show, from_networkx # type: ignore
from bokeh.models import Circle, MultiLine, HoverTool, LinearColorMapper, ColorBar, WheelZoomTool # type: ignore
from bokeh.tile_providers import get_provider, Vendors # type: ignore
from bokeh.palettes import Category10, Category20 # type: ignore
from bokeh.io.export import export_png # type: ignore
from pyproj import Transformer # type: ignore
from bokeh.models import GMapOptions # type: ignore
from bokeh.plotting import gmap # type: ignore
import networkx as nx # type: ignore

def plot_graph(G, space="L", back_map=False, MAPS_API_KEY=None, color_by="", edge_color_by="", export_name=""):
    if back_map == "GMAPS":
        first_node = next(iter(G.nodes(data=True)))
        map_options = GMapOptions(lat=first_node[1]["lat"], lng=first_node[1]["lon"], map_type="roadmap", zoom=11)
        p = gmap(MAPS_API_KEY, map_options)
    else:
        p = figure(height=600, width=950, toolbar_location='below', tools="pan, wheel_zoom, box_zoom, reset, save")

    # Build node position dict
    pos_dict = {}
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    for i, d in G.nodes(data=True):
        if back_map == "OSM":
            x2, y2 = transformer.transform(float(d["lon"]), float(d["lat"]))
        else:
            x2, y2 = float(d["lon"]), float(d["lat"])
        pos_dict[i] = (x2, y2)

    graph = from_networkx(G, layout_function=pos_dict)

    # Hover tools
    node_hover_tool = HoverTool(tooltips=[("index", "@index"), ("name", "@name")], renderers=[graph.node_renderer])
    edge_tooltips = [("duration_avg", "@duration_avg")] if space == "L" else [("avg_wait", "@avg_wait")]
    hover_edges = HoverTool(tooltips=edge_tooltips, renderers=[graph.edge_renderer], line_policy="interp")
    p.add_tools(node_hover_tool, hover_edges)

    # Node coloring
    if color_by and all(color_by in d for _, d in G.nodes(data=True)):
        mapper = LinearColorMapper(palette="RdYlGn11", low=min(nx.get_node_attributes(G, color_by).values()), high=max(nx.get_node_attributes(G, color_by).values()))
        graph.node_renderer.glyph = Circle(size=7, fill_color={'field': color_by, 'transform': mapper})
    else:
        graph.node_renderer.glyph = Circle(size=7)

    # Edge coloring
    if edge_color_by and all(edge_color_by in d for _, _, d in G.edges(data=True)):
        edge_vals = [d[edge_color_by] for _, _, d in G.edges(data=True)]
        mapper = LinearColorMapper(palette="RdYlGn11", low=min(edge_vals), high=max(edge_vals))
        graph.edge_renderer.glyph = MultiLine(line_width=4, line_alpha=0.5, line_color={'field': edge_color_by, 'transform': mapper})
        color_bar = ColorBar(color_mapper=mapper, label_standoff=12, border_line_color=None, location=(0, 0))
        p.add_layout(color_bar, "right")
    else:
        graph.edge_renderer.glyph = MultiLine(line_width=4, line_alpha=0.5)

    graph.node_renderer.selection_glyph = Circle(fill_color='blue')
    graph.node_renderer.hover_glyph = Circle(fill_color='red')

    p.toolbar.active_scroll = p.select_one(WheelZoomTool)

    if space == "P":
        graph.edge_renderer.selection_glyph = MultiLine(line_color='black', line_width=5)
        graph.edge_renderer.hover_glyph = MultiLine(line_color='black', line_width=10)
    else:
        graph.edge_renderer.selection_glyph = MultiLine(line_color='blue', line_width=5)
        graph.edge_renderer.hover_glyph = MultiLine(line_color='red', line_width=5)

    p.renderers.append(graph)

    if back_map == "OSM":
        p.add_tile(get_provider(Vendors.CARTODBPOSITRON))

    if export_name:
        export_png(p, filename=export_name + ".png")
    else:
        show(p)

def plot_nodes_highlight(G, nodes, back_map="OSM", MAPS_API_KEY=None):
    """
    Plot the graph with given nodes highlighted in different colors.

    Args:
        G (nx.Graph): Graph with 'lat' and 'lon' node attributes.
        nodes (list): Node IDs to highlight.
        back_map (str): "OSM", "GMAPS", or None.
        MAPS_API_KEY (str, optional): Required if back_map == "GMAPS".
    """
    if not isinstance(nodes, (list, tuple, set)):
        nodes = [nodes]

    if back_map == "GMAPS":
        first_node = next(iter(G.nodes(data=True)))
        map_options = GMapOptions(lat=first_node[1]["lat"],
                                  lng=first_node[1]["lon"],
                                  map_type="roadmap",
                                  zoom=11)
        p = gmap(MAPS_API_KEY, map_options)
    else:
        p = figure(height=600, width=950, toolbar_location="below",
                   tools="pan, wheel_zoom, box_zoom, reset, save")

    # Build node position dictionary
    pos_dict = {}
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    for i, d in G.nodes(data=True):
        if back_map == "OSM":
            x2, y2 = transformer.transform(float(d["lon"]), float(d["lat"]))
        else:
            x2, y2 = float(d["lon"]), float(d["lat"])
        pos_dict[i] = (x2, y2)

    graph = from_networkx(G, layout_function=pos_dict)

    # Default node/edge styling
    graph.node_renderer.glyph = Circle(size=7, fill_color="gray")
    graph.edge_renderer.glyph = MultiLine(line_width=2, line_alpha=0.5)

    # Choose color palette
    palette = Category20[20] if len(nodes) > 10 else Category10[10]

    # Highlight each requested node
    for idx, node in enumerate(nodes):
        if node in pos_dict:
            color = palette[idx % len(palette)]
            x, y = [pos_dict[node][0]], [pos_dict[node][1]]
            p.circle(x=x, y=y, size=15, color=color, legend_label=f"Node {node}")
        else:
            print(f"Node {node} not found in graph")

    # Add hover tools
    node_hover_tool = HoverTool(tooltips=[("index", "@index"), ("name", "@name")],
                                renderers=[graph.node_renderer])
    p.add_tools(node_hover_tool)

    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    p.renderers.append(graph)

    if back_map == "OSM":
        p.add_tile(get_provider(Vendors.CARTODBPOSITRON))

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    show(p)

def plot_efficiency_results(percent_remaining, efficiencies, title="Impact of Node Removal on Network Efficiency (Normalized)"):
    """
    Plots the change in normalized efficiency as nodes are removed.

    Parameters:
    - num_removed: List of number of nodes removed
    - efficiencies: Corresponding list of normalized efficiencies
    - title: Plot title
    """
    plt.figure(figsize=(6, 4))
    plt.plot(percent_remaining, efficiencies, marker='o')
    plt.xlabel("Percentage Remaining")
    plt.ylabel("Normalized Efficiency")
    plt.title(title)
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

def plot_efficiency_vs_custom_nodes(efficiencies, removed_nodes, title="Normalized Efficiency vs Removed Nodes"):
    """
    Plots normalized efficiency vs the actual node IDs removed, including initial efficiency.

    Parameters:
    - efficiencies: List of efficiency values at each removal step (first value = initial graph)
    - removed_nodes: List of node IDs removed in order
    - title: Plot title
    """
    # X-axis: initial graph + removed nodes
    x_axis = ["Initial"] + removed_nodes
    normalized_eff = [eff / efficiencies[0] for eff in efficiencies]

    plt.figure(figsize=(6, 4))
    plt.plot(x_axis, normalized_eff, marker='o')
    plt.xlabel("Node Removed")
    plt.ylabel("Normalized Efficiency")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_efficiency_with_node_labels(efficiencies, node_names, title="Impact of Node Removal on Network Efficiency (Normalized)"):
    """
    Plot normalized efficiency vs node removals, using node names as x-axis labels.

    Parameters:
    - percent_remaining: List of percentage of nodes remaining
    - efficiencies: Corresponding list of normalized efficiencies
    - node_names: List of node names in order of removal (length should be len(efficiencies) - 1)
    - title: Plot title
    """

    plt.figure(figsize=(10, 5))

    # x positions: include one extra for the starting point (no node removed)
    x_positions = list(range(len(efficiencies)))

    plt.plot(x_positions, efficiencies, marker='o')

    # Create x-axis labels: first is "Start" or "None", then the removed node names
    x_labels = ["Full Graph"] + node_names

    plt.xticks(ticks=x_positions, labels=x_labels, rotation=90, ha='center')

    plt.xlabel("Removed Nodes")
    plt.ylabel("Normalized Efficiency")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_efficiency_results_from_batch(row):
    """
    Plot the efficiency drop across node removals for a single subgraph.

    Parameters:
        row (pd.Series): A row from the DataFrame containing keys:
            - 'num_nodes': total nodes in subgraph
            - 'efficiency_after_each_removal': list of normalized efficiency after each node removal
    """
    total_nodes = row['num_nodes']
    efficiencies = [1.0] + row['efficiency_after_each_removal']  # efficiency before any removal + after each removal
    
    num_removed = list(range(len(efficiencies)))  # 0, 1, 2, ... nodes removed
    percent_remaining = [100 * (total_nodes - n) / total_nodes for n in num_removed]

    plot_efficiency_results(percent_remaining, efficiencies)


def compute_avg_runtime_by_num_nodes(df_results):
    """
    Compute the average and total runtime, and total number of nodes removed for subgraphs grouped by number of nodes.

    Parameters:
        df_results (pd.DataFrame): DataFrame with columns:
            - 'num_nodes': int, number of nodes in the subgraph
            - 'runtime_seconds': float, total runtime for removals on the subgraph
            - 'removed_nodes': list, nodes removed from the subgraph

    Returns:
        pd.DataFrame: DataFrame with columns:
            - 'num_nodes': number of nodes in each subgraph
            - 'total_nodes_removed': total number of nodes removed
            - 'avg_runtime_seconds': average runtime (in seconds)
            - 'total_runtime_seconds': total runtime for that graph size
    """
    # Add a column for number of removed nodes per row
    df_results["pct_nodes_removed"] = df_results["removed_entities"].apply(len)

    # Group and aggregate
    grouped = df_results.groupby("num_nodes").agg(
        avg_runtime_removal_seconds=("runtime_seconds", "mean"),
        total_runtime_removal_seconds=("runtime_seconds", "sum")
    ).reset_index()

    return grouped

def plot_removal_time_vs_steps(row):
    """
    Plot cumulative runtime and individual removal times against number of node removals for a single subgraph,
    with two side-by-side subplots. Also displays a table of removed nodes and corresponding removal times.
    
    Parameters:
    row (pd.Series): Row from df_results containing 'removal_times' and 'removed_nodes'.
    """
    if "removal_times" not in row or not row["removal_times"]:
        print("No timing data available for this row.")
        return

    individual_times = row["removal_times"]
    cumulative_times = np.cumsum(individual_times)
    steps = list(range(1, len(individual_times) + 1))
    
    # Display tabular data
    df = pd.DataFrame({
        "Node Removed": row["removed_entities"],
        "Time Elapsed (s)": individual_times
    })
    display(df) # type: ignore

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: time line plot
    ax1.plot(steps, individual_times, marker='o', color='b')
    ax1.set_title(f"Removal Time\nGraph Size {row['num_nodes']} Index {row['graph_index']}")
    ax1.set_xlabel("Node Removal Step")
    ax1.set_ylabel("Time per Removal (seconds)")
    ax1.grid(True)

    # Right: individual removal time bar plot
    ax2.bar(steps, individual_times, color='orange', alpha=0.7)
    ax2.set_title("Individual Removal Time per Node")
    ax2.set_xlabel("Node Removal Step")
    ax2.set_ylabel("Time per Removal (seconds)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_efficiency_decay(df, graphs_per_group=2, color_map=None, title="Efficiency Decay Across Subgraphs"):
    """
    Plots efficiency decay lines for selected subgraphs grouped by num_nodes.

    Parameters:
        df (pd.DataFrame): DataFrame with 'num_nodes' and 'eff_after_' columns
        graphs_per_group (int): Number of graphs per num_nodes category
        color_map (str or dict): Name of a Matplotlib colormap or a dict of num_nodes to color
        title (str): Plot title
    """
    unique_nodes = sorted(df["num_nodes"].unique())

    # Handle colormap input
    if isinstance(color_map, str):
        colormap = cm.get_cmap(color_map, len(unique_nodes))
        color_map = {num_nodes: colormap(i) for i, num_nodes in enumerate(unique_nodes)}
    elif color_map is None:
        colormap = cm.get_cmap('tab10', len(unique_nodes))
        color_map = {num_nodes: colormap(i) for i, num_nodes in enumerate(unique_nodes)}
    elif not isinstance(color_map, dict):
        raise ValueError("color_map must be a string, dictionary, or None")

    # Subset the DataFrame
    df_subset = df.groupby("num_nodes", group_keys=False).head(graphs_per_group)

    eff_cols = [col for col in df_subset.columns if col.startswith("eff_after_")]
    plt.figure(figsize=(10, 5))
    plotted_labels = set()

    for _, row in df_subset.iterrows():
        num_nodes = row["num_nodes"]
        color = color_map.get(num_nodes, 'gray')
        eff_values = [row[col] for col in eff_cols if not pd.isna(row[col])]
        x = list(range(len(eff_values)))

        label = f"{num_nodes}" if num_nodes not in plotted_labels else None
        plt.plot(x, eff_values, color=color, label=label)
        plotted_labels.add(num_nodes)

    plt.title(title)
    plt.xlabel("Nodes Removed")
    plt.ylabel("Normalized Efficiency")
    plt.grid(True)
    plt.legend(title="Number of Nodes")
    plt.tight_layout()
    plt.show()

def remove_node_edges_and_plot(G, nodes):
    """
    Removes all edges connected to the specified list of nodes from the graph.
    Prints a message if a node does not exist in the graph.

    Parameters:
        G (networkx.Graph): The graph to modify (passed by reference).
        nodes (list): List of nodes whose edges will be removed.

    Returns:
        networkx.Graph: The modified graph with specified edges removed.
    """
    if not isinstance(nodes, list):
        nodes = [nodes]  # Ensure single node inputs also work

    edges_to_remove = []
    
    for node in nodes:
        if G.has_node(node):
            if G.is_directed():
                edges_to_remove += list(G.in_edges(node)) + list(G.out_edges(node))
            else:
                edges_to_remove += list(G.edges(node))
        else:
            print(f"Node '{node}' not found in the graph. Skipping.")

    G.remove_edges_from(edges_to_remove)
    return G



def plot_runtime_comparison(runtimes, subgraph_sizes, versions, colors, bar_width=0.2, group_gap=0.3):
    for size in subgraph_sizes:
        try:
            num_subgraphs = len(next(iter(runtimes.values()))[size])
        except (KeyError, StopIteration):
            print(f"No data available for subgraph size {size}")
            continue

        total_versions = len(versions)
        group_width = total_versions * bar_width + group_gap
        x = np.arange(num_subgraphs) * group_width  # insert space between subgraph groups
        offsets = np.linspace(
            -bar_width * (total_versions - 1) / 2,
            bar_width * (total_versions - 1) / 2,
            total_versions
        )

        plt.figure(figsize=(10, 5))

        for i, version in enumerate(versions):
            if size not in runtimes.get(version, {}):
                print(f"Skipping version {version} for subgraph size {size} (data missing)")
                continue

            y = runtimes[version][size]
            positions = x + offsets[i]
            plt.bar(positions, y, width=bar_width, color=colors.get(version, "gray"), label=version)

        plt.title(f"Runtime Comparison for Subgraph Size {size} at 50% Node Removal")
        plt.xlabel("Subgraph Index")
        plt.ylabel("Runtime (seconds)")
        plt.xticks(x, [f"{i+1}" for i in range(num_subgraphs)])
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()


def num_route_dir_pairs_with_density(L):
    """
    Compute the number of unique route-direction pairs in subgraph L
    and the density defined as pairs / number of nodes.

    Args:
        L: networkx graph with edges having 'route_I_counts' and optionally 'direction_id' attribute

    Returns:
        tuple: (num_pairs, density)
               num_pairs (int): Number of unique (route, direction) pairs found on edges.
               density (float): num_pairs divided by number of nodes in L.
    """
    route_dir_pairs = set()

    for _, _, edge_data in L.edges(data=True):
        route_counts = edge_data.get('route_I_counts', {})
        dir_dict = edge_data.get('direction_id', {})

        for route in route_counts.keys():
            if dir_dict:
                for direction in dir_dict.keys():
                    route_dir_pairs.add((route, direction))
            else:
                route_dir_pairs.add((route, None))

    num_pairs = len(route_dir_pairs)
    num_nodes = L.number_of_nodes()
    density = num_pairs / num_nodes if num_nodes > 0 else 0

    return num_pairs, density

def sort_subgraphs_dict_by_route_dir_pairs(subgraphs_dict):
    sorted_subgraphs_dict = {}
    for size, sg_list in subgraphs_dict.items():
        # Sort subgraphs in descending order by density (route-dir pairs / number of nodes)
        sorted_sgs = sorted(
            sg_list,
            key=lambda g: num_route_dir_pairs_with_density(g)[1],  # density is at index 1
            reverse=True
        )
        sorted_subgraphs_dict[size] = sorted_sgs
    return sorted_subgraphs_dict

def plot_runtime_bars(runtimes, subgraph_sizes, versions, colors, bar_width=0.2):
    for size in subgraph_sizes:
        try:
            num_subgraphs = len(next(iter(runtimes.values()))[size])
        except (KeyError, StopIteration):
            print(f"Subgraph size {size} not available in runtimes.")
            continue

        x = np.arange(num_subgraphs)
        total_versions = len(versions)
        offsets = np.linspace(
            -bar_width * (total_versions - 1) / 2,
            bar_width * (total_versions - 1) / 2,
            total_versions
        )

        plt.figure(figsize=(8, 5))

        for i, version in enumerate(versions):
            if size not in runtimes.get(version, {}):
                print(f"Version {version} does not contain subgraph size {size}. Skipping.")
                continue

            y = runtimes[version][size]
            if len(y) != num_subgraphs:
                print(f"Length mismatch for version {version} at size {size}. Skipping.")
                continue

            plt.bar(x + offsets[i], y, width=bar_width, color=colors.get(version, "gray"), label=version)

        plt.title(f"Runtime Comparison for Subgraph Size {size} at 40% Node Removal")
        plt.xlabel("Subgraph Index")
        plt.ylabel("Runtime (seconds)")
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

def plot_runtime_vs_density_scatter(runtimes, sorted_subgraphs, versions, subgraph_sizes, colors, density_func):
    """
    Plots runtime vs density scatter plots for specified versions and subgraph sizes.

    Parameters:
        runtimes (dict): Nested dict of runtimes[version][subgraph_size] = list of runtimes.
        sorted_subgraphs (dict): Dict of subgraph_size -> list of graphs.
        versions (list): List of version keys to include in the plot.
        subgraph_sizes (list): List of subgraph sizes to plot.
        colors (dict): Mapping from version to color.
        density_func (callable): Function to compute density metric. Must return (score, ...) where score is numeric.
    """
    for size in subgraph_sizes:
        plt.figure(figsize=(8, 5))
        
        for version in versions:
            graphs = sorted_subgraphs.get(size, [])
            densities = []
            runtimes_list = []
            
            for i, graph in enumerate(graphs, start=1):
                score, _ = density_func(graph)  # Assumes it returns a tuple (score, extra)
                num_nodes = graph.number_of_nodes()
                density = score / num_nodes if num_nodes > 0 else 0

                try:
                    runtime = runtimes[version][size][i - 1]
                except (KeyError, IndexError):
                    runtime = None
                
                if runtime is not None:
                    densities.append(density)
                    runtimes_list.append(runtime)
            
            if densities and runtimes_list:
                plt.scatter(densities, runtimes_list, color=colors.get(version, "gray"), label=version, alpha=0.7)
            else:
                print(f"No data points to plot for {version} size {size}")
        
        plt.title(f"Runtime vs Density for Subgraph Size {size}")
        plt.xlabel("Density (Route-Direction pairs / Number of Nodes)")
        plt.ylabel("Runtime (seconds)")
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

def plot_efficiency_results_multi(efficiency_data, size, versions=None):
    """
    Plots efficiency curves for specified versions and subgraph size.

    Parameters:
    - efficiency_data: dict from get_efficiency_curves output:
        { version_label: { size: [ { 'curve': [...], 'seed': int, ... }, ... ] } }
    - seeds: List of seed values used
    - size: Integer size of the subgraph to plot
    - versions: Optional list of version labels to plot (e.g. ['v0', 'v4']); if None, plots all
    """
    if versions is None:
        versions = sorted(efficiency_data.keys())

    plt.figure(figsize=(6 * len(versions), 5))

    for i, label in enumerate(versions, start=1):
        plt.subplot(1, len(versions), i)

        runs = efficiency_data.get(label, {}).get(size, [])
        if not runs:
            plt.title(f'{label} - Size {size} (no data)')
            continue

        for idx, run in enumerate(runs):
            curve = run['curve']
            seed = run.get('seed', 'unknown')
            plt.plot(curve, label=f'Seed {seed}, Run {idx + 1}')

        plt.title(f'{label} - Size {size}')
        plt.xlabel('Nodes removed')
        plt.ylabel('Efficiency')
        plt.legend(fontsize='small', loc='best')
        plt.grid(True)

    plt.tight_layout(rect=[0, 0, 0.9, 0.75])
    plt.show()

def analyze_runtime_improvement(runtimes, from_version='v1', to_version='v5'):
    """
    Computes and plots the Pareto curve of runtime improvements from one version to another.
    
    Parameters:
    - runtimes (dict): Nested dict of runtimes[version][subgraph_size] = list of runtimes
    - from_version (str): Version to compare from
    - to_version (str): Version to compare to
    """
    improvements = []
    for size in runtimes.get(from_version, {}):
        v_from = runtimes[from_version].get(size, [])
        v_to = runtimes.get(to_version, {}).get(size, [])
        for r1, r2 in zip(v_from, v_to):
            if r1 != 0:
                improvements.append((r1 - r2) / r1)

    if not improvements:
        print("No valid improvement data found.")
        return

    average_improvement = sum(improvements) / len(improvements)
    print(f"Average improvement from {from_version} to {to_version}: {average_improvement:.2%}")

    sorted_improvements = sorted(improvements, reverse=True)
    cum_percent = [i / len(sorted_improvements) * 100 for i in range(len(sorted_improvements))]

    plt.figure(figsize=(6, 4))
    plt.plot(cum_percent, sorted_improvements, marker='o')
    plt.xlabel("Cumulative Percentage of Subgraphs")
    plt.ylabel(f"Relative Runtime Improvement ({from_version} to {to_version})")
    plt.title(f"Pareto Curve of Runtime Improvement from {from_version} to {to_version}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_efficiency_from_loaded_df(df, num_nodes):
    """
    Plot efficiency degradation from a loaded removal results DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame loaded from CSV, containing:
            - 'normalized_efficiency'
        num_nodes (int): Total number of nodes in the original graph.
    """
    efficiencies = df['normalized_efficiency'].tolist()
    if efficiencies[0] == 1.0:
        efficiency_after_each_removal = efficiencies[1:]
    else:
        efficiency_after_each_removal = efficiencies

    mock_row = pd.Series({
        'num_nodes': num_nodes,
        'efficiency_after_each_removal': efficiency_after_each_removal
    })

    plot_efficiency_results_from_batch(mock_row)


def plot_multiple_efficiency_runs(results_dir, color='blue', title='Efficiency Degradation Across Multiple Runs', legend=False):
    """
    Plot individual efficiency runs, then a separate plot showing mean efficiency ± std deviation.
    Works for filenames containing either 'nodesX' or 'edgesX' at the end before .csv.
    """
    results_dir = Path(results_dir)
    csv_files = [f for f in results_dir.iterdir() if f.suffix == '.csv']

    if not csv_files:
        print("No CSV files found in the directory.")
        return

    # -------- FIRST PLOT: Individual Runs --------
    plt.figure(figsize=(10, 6))

    all_curves = []
    common_x = np.linspace(0, 100, 100)  # Common 0–100% scale for interpolation

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        filename = csv_file.stem

        match = re.search(r'(nodes|edges)(\d+)', filename)
        if match:
            total_nodes = int(match.group(2))
        else:
            print(f"Warning: Could not extract number of nodes/edges from filename '{filename}', skipping this file.")
            continue

        efficiencies = df['normalized_efficiency'].tolist()
        if efficiencies[0] == 1.0:
            efficiencies = efficiencies[1:]

        efficiencies = [1.0] + efficiencies
        num_removed = list(range(len(efficiencies)))
        percent_remaining = [100 * (total_nodes - n) / total_nodes for n in num_removed]

        # Interpolate to common x for averaging later
        interp_eff = np.interp(common_x, percent_remaining[::-1], efficiencies[::-1])
        all_curves.append(interp_eff)

        # Plot individual run
        if legend:
            plt.plot(percent_remaining, efficiencies, label=filename, alpha=0.5)
        else:
            plt.plot(percent_remaining, efficiencies, color=color, alpha=0.5)

    plt.xlabel("Percentage of Nodes Remaining")
    plt.ylabel("Normalized Efficiency")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.gca().invert_xaxis()
    if legend:
        plt.legend()
    plt.show()

    # -------- SECOND PLOT: Mean ± Std with Data Points --------
    all_curves = np.array(all_curves)
    mean_eff = np.mean(all_curves, axis=0)
    std_eff = np.std(all_curves, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(common_x, mean_eff, color='black', linewidth=2, marker='o', markersize=4, label='Mean Efficiency')
    plt.fill_between(common_x, mean_eff - std_eff, mean_eff + std_eff, color='gray', alpha=0.3, label='±1 SD')

    plt.xlabel("Percentage of Nodes Remaining")
    plt.ylabel("Normalized Efficiency")
    plt.title(title + " (Mean ± Std Deviation)")
    plt.grid(True)
    plt.tight_layout()
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()


def plot_efficiency_comparison_single(run_configs, title="", xlim=None):
    """
    Plot efficiency curves for multiple runs on the same graph,
    shading the area above each curve and calculating the area value.

    Args:
        run_configs (list of dict): Each dict must have keys:
            'fil' (Path or str to CSV file), 'color', 'label'
        title (str): Plot title
        xlim (tuple, optional): (min_x, max_x) range for zoom (percent remaining).
                                Tuple order does not matter; output is always descending.
    """
    plt.figure(figsize=(10, 6))
    areas_above = {}

    # Decide whether we're working with nodes or edges based on the first file
    filename0 = Path(run_configs[0]['fil']).name
    if "_nodes" in filename0:
        keyword = "_nodes"
    elif "_edges" in filename0:
        keyword = "_edges"
    else:
        raise ValueError("Filenames must contain either '_nodes' or '_edges'")

    for cfg in run_configs:
        df = pd.read_csv(cfg['fil'])
        efficiencies = df['normalized_efficiency'].tolist()

        # Ensure starting point at 1.0
        if efficiencies[0] == 1.0:
            efficiencies = efficiencies[1:]
        efficiencies = [1.0] + efficiencies

        # Extract total count from filename
        filename = Path(cfg['fil']).name
        if keyword not in filename:
            raise ValueError(f"Inconsistent file naming: expected {keyword} in {filename}")
        num_str = filename.split(keyword)[-1].replace(".csv", "")
        total_count = int(num_str)

        num_removed = list(range(len(efficiencies)))
        percent_remaining = [100 * (total_count - n) / total_count for n in num_removed]

        # Calculate area above curve
        gap_above = [1 - x for x in efficiencies]
        area_above = integrate.trapezoid(gap_above, dx=100 / total_count)
        areas_above[cfg['label']] = area_above

        # Plot smooth line
        plt.plot(
            percent_remaining,
            efficiencies,
            color=cfg['color'],
            label=cfg['label']
        )

        # Overlay scatter points
        plt.scatter(
            percent_remaining,
            efficiencies,
            color=cfg['color'],
            s=15,        # point size
            alpha=0.7
        )

        # Shade area above curve
        plt.fill_between(percent_remaining, efficiencies, 1.0, color=cfg['color'], alpha=0.3)


    # Apply zoom if specified
    if xlim:
        plt.xlim(xlim)

    # Always ensure descending x-axis
    if plt.gca().get_xlim()[0] < plt.gca().get_xlim()[1]:
        plt.gca().invert_xaxis()

    plt.xlabel(f"Percentage of {keyword.strip('_').capitalize()} Remaining")
    plt.ylabel("Normalized Efficiency")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print area results
    for label, area in areas_above.items():
        print(f"Area above curve ({label}): {area:.4f}")

    return areas_above


def plot_efficiency_comparison_multi(run_configs, title='Efficiency Comparison', xlim=None):
    """
    Plot efficiency curves from multiple run directories (countries) side-by-side:
    Left plot: individual curves colored by group with legend.
    Right plot: average curve with shaded area for each group.

    Args:
        run_configs (list of dict): Each dict must have keys 'dir', 'color', and 'label'.
        title (str): Title for the whole figure.
        xlim (tuple, optional): (min_x, max_x) range to zoom in on x-axis (percent remaining).
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    ax1, ax2 = axs
    legend_elements = []
    plotted_left = False
    plotted_right = False

    # plt.rcParams.update({
    #     'axes.titlesize': 22,      # Title font size for subplots
    #     'axes.labelsize': 24,      # X and Y label size
    #     'legend.fontsize': 16,     # Legend font size
    #     'figure.titlesize': 18     # Figure-level title font size
    # })

    # --- LEFT PLOT: individual curves ---
    for config in run_configs:
        directory = Path(config['dir'])
        color = config['color']
        label = config['label']

        csv_files = [f for f in directory.iterdir() if f.suffix == '.csv']

        for csv_file in csv_files:
            try:
                filename = csv_file.name

                # --- Support for both nodes and edges ---
                if "_nodes" in filename:
                    num_str = filename.split("_nodes")[-1].replace(".csv", "")
                elif "_edges" in filename:
                    num_str = filename.split("_edges")[-1].replace(".csv", "")
                else:
                    print(f"Warning: Could not extract number from '{filename}', skipping.")
                    continue

                total_elements = int(num_str)

                df = pd.read_csv(csv_file)
                if 'normalized_efficiency' not in df.columns:
                    print(f"Warning: Missing 'normalized_efficiency' in '{filename}', skipping.")
                    continue

                efficiencies = df['normalized_efficiency'].tolist()
                if efficiencies[0] == 1.0:
                    efficiencies = efficiencies[1:]
                efficiencies = [1.0] + efficiencies

                num_removed = list(range(len(efficiencies)))
                percent_remaining = [100 * (total_elements - n) / total_elements for n in num_removed]

                ax1.plot(percent_remaining, efficiencies, color=color)
                plotted_left = True

            except Exception as e:
                print(f"Skipping {filename}: {e}")
                continue

        legend_elements.append(Line2D([0], [0], color=color, lw=2, label=label))

    ax1.set_xlabel("Percentage Remaining")
    ax1.set_ylabel("Normalized Efficiency")
    ax1.set_title("Individual Efficiency Curves")
    ax1.grid(True)
    ax1.invert_xaxis()
    if plotted_left:
        ax1.legend(handles=legend_elements)

    # --- RIGHT PLOT: average curves with shaded area ---
    for config in run_configs:
        directory = Path(config['dir'])
        color = config['color']
        label = config['label']

        csv_files = [f for f in directory.iterdir() if f.suffix == '.csv']
        all_efficiencies = []
        all_percent_remaining = []

        for csv_file in csv_files:
            try:
                filename = csv_file.name

                # --- Support for both nodes and edges ---
                if "_nodes" in filename:
                    num_str = filename.split("_nodes")[-1].replace(".csv", "")
                elif "_edges" in filename:
                    num_str = filename.split("_edges")[-1].replace(".csv", "")
                else:
                    print(f"Warning: Could not extract number from '{filename}', skipping.")
                    continue

                total_elements = int(num_str)

                df = pd.read_csv(csv_file)
                if 'normalized_efficiency' not in df.columns:
                    print(f"Warning: Missing 'normalized_efficiency' in '{filename}', skipping.")
                    continue

                efficiencies = df['normalized_efficiency'].tolist()
                if efficiencies[0] == 1.0:
                    efficiencies = efficiencies[1:]
                efficiencies = [1.0] + efficiencies

                num_removed = list(range(len(efficiencies)))
                percent_remaining = [100 * (total_elements - n) / total_elements for n in num_removed]

                all_efficiencies.append(efficiencies)
                all_percent_remaining.append(percent_remaining)

            except Exception as e:
                print(f"Skipping {filename}: {e}")
                continue

        if not all_efficiencies:
            continue

        min_len = min(len(e) for e in all_efficiencies)
        truncated_efficiencies = [e[:min_len] for e in all_efficiencies]
        truncated_percent_remaining = all_percent_remaining[0][:min_len]

        mean_efficiency = np.mean(truncated_efficiencies, axis=0)

        ax2.plot(truncated_percent_remaining, mean_efficiency, color=color, label=label)
        ax2.fill_between(truncated_percent_remaining, mean_efficiency, 1.0, color=color, alpha=0.3)
        plotted_right = True

    ax2.set_xlabel("Percentage Remaining")
    ax2.set_ylabel("Normalized Efficiency")
    ax2.set_title("Average Efficiency Curves")
    ax2.grid(True)
    ax2.invert_xaxis()
    if plotted_right:
        ax2.legend()

    if xlim:
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_average_efficiency_with_area(results_dir):
    """
    Load all CSV files in results_dir, compute average efficiency curve,
    plot the average (red) line with red circles at data points and shade the upper area.
    Works for filenames containing either 'nodesX' or 'edgesX' before .csv.
    Also computes and prints statistics over individual curve areas.

    Args:
        results_dir (Path or str): Directory containing CSV files with 'normalized_efficiency' column.

    Returns:
        float: Area above the average efficiency curve
        pd.DataFrame: DataFrame of individual areas
    """
    results_dir = Path(results_dir)
    csv_files = [f for f in results_dir.glob("*.csv")]
    
    all_efficiencies = []
    all_percent_remaining = []
    individual_areas = []
    node_counts = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            efficiencies = df['normalized_efficiency'].tolist()

            if efficiencies[0] == 1.0:
                efficiencies = efficiencies[1:]
            efficiencies = [1.0] + efficiencies

            # Extract number of nodes or edges from filename
            filename_stem = csv_file.stem
            match = re.search(r'(nodes|edges)(\d+)', filename_stem)
            if match:
                total_nodes = int(match.group(2))
            else:
                print(f"Skipping {csv_file.name}: Could not extract number of nodes/edges.")
                continue

            node_counts.append(total_nodes)
            num_removed = list(range(len(efficiencies)))
            percent_remaining = [100 * (total_nodes - n) / total_nodes for n in num_removed]

            all_efficiencies.append(efficiencies)
            all_percent_remaining.append(percent_remaining)

            # Compute individual area above curve
            gap_above = [1 - x for x in efficiencies]
            area_above = integrate.trapezoid(gap_above, dx=100 / total_nodes)
            individual_areas.append(area_above)

        except Exception as e:
            print(f"Skipping {csv_file.name}: {e}")

    if not all_efficiencies:
        raise ValueError("No valid CSV files with 'normalized_efficiency' found.")

    # Find the minimum length to align all runs
    min_len = min(len(e) for e in all_efficiencies)
    truncated_efficiencies = [e[:min_len] for e in all_efficiencies]
    truncated_percent_remaining = all_percent_remaining[0][:min_len]

    # Compute average efficiency
    mean_efficiency = np.mean(truncated_efficiencies, axis=0)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(truncated_percent_remaining, mean_efficiency, color='red')
    plt.scatter(truncated_percent_remaining, mean_efficiency, color='red', edgecolors='none', zorder=5)
    plt.fill_between(truncated_percent_remaining, mean_efficiency, 1.0, color='red', alpha=0.3)

    plt.xlabel("Percentage of Nodes Remaining")
    plt.ylabel("Normalized Efficiency")
    plt.title("Average Efficiency Degradation (Red Line and Shaded Area)")
    plt.grid(True)
    plt.tight_layout()
    plt.gca().invert_xaxis()
    plt.show()

    # Compute area above average curve
    gap_above_avg = [1 - x for x in mean_efficiency]
    dx = (truncated_percent_remaining[0] - truncated_percent_remaining[-1]) / (len(gap_above_avg) - 1)
    area_above_avg = integrate.trapezoid(gap_above_avg, dx=dx)

    print(f"Area above average efficiency line: {area_above_avg:.4f}\n")

    # Individual areas table
    df_areas = pd.DataFrame({
        'File': [f.name for f in csv_files[:len(individual_areas)]],
        'Area Above Curve': individual_areas
    })
    print(df_areas.to_string(index=False))

    # Summary statistics
    print("\nStatistics over area above efficiency curves:")
    print(f"Mean   : {np.mean(individual_areas):.4f}")
    print(f"Median : {np.median(individual_areas):.4f}")
    print(f"Min    : {np.min(individual_areas):.4f}")
    print(f"Max    : {np.max(individual_areas):.4f}")
    print(f"Std Dev: {np.std(individual_areas):.4f}")

    return area_above_avg, df_areas


def plot_efficiency_with_node_labels_from_df(df, title="Network Efficiency over Node Removals"):
    """
    Plot normalized efficiency degradation with red line, points, shaded area,
    and return area above the curve.

    Args:
        df (pd.DataFrame): Must include 'normalized_efficiency' and 'removed_node_name'
        title (str): Plot title

    Returns:
        float: Area above the efficiency curve
    """
    efficiencies = df['normalized_efficiency'].tolist()
    node_labels = df['removed_node_names'].tolist()

    # Prepend full graph efficiency = 1
    if efficiencies[0] != 1.0:
        efficiencies = [1.0] + efficiencies
        node_labels = [''] + node_labels  # Empty label for initial full graph

    x_labels = ['Full Graph'] + node_labels[1:]
    x_positions = list(range(len(efficiencies)))

    plt.figure(figsize=(12, 6))
    plt.plot(x_positions, efficiencies, color='red', marker='o', markerfacecolor='red', linewidth=2)
    plt.fill_between(x_positions, efficiencies, 1.0, color='red', alpha=0.2)
    plt.xticks(ticks=x_positions, labels=x_labels, rotation=45, ha='center')
    plt.xlabel("Removed Nodes")
    plt.ylabel("Normalized Efficiency")
    plt.title(title)
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # Compute area above the curve (gap between full efficiency and actual)
    gap_above = [1 - x for x in efficiencies]
    area_above = integrate.trapezoid(gap_above, dx=1)
    print(f"Area above average efficiency line: {area_above:.4f}\n")

    return area_above