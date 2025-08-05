import matplotlib.pyplot as plt # type: ignore
import matplotlib.cm as cm # type: ignore
import numpy as np 
import pandas as pd # type: ignore
from scipy import integrate # type: ignore
from pathlib import Path

from bokeh.plotting import figure, show, from_networkx # type: ignore
from bokeh.models import Circle, MultiLine, HoverTool, LinearColorMapper, ColorBar, WheelZoomTool # type: ignore
from bokeh.tile_providers import get_provider, Vendors # type: ignore
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

    plt.xticks(ticks=x_positions, labels=x_labels, rotation=0, ha='center')

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

def remove_node_edges_and_plot(G, node):
    """
    Removes all edges connected to the specified node from the graph and plots the result.

    Parameters:
        G (networkx.Graph): The graph to modify (passed by reference).
        node: The node whose edges will be removed.
        back_map (str): Parameter passed to plot_graph function for background map.
    """
    if G.is_directed():
        edges_to_remove = list(G.in_edges(node)) + list(G.out_edges(node))
    else:
        edges_to_remove = list(G.edges(node))

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


def plot_multiple_efficiency_runs(results_dir):
    """
    Load all CSV files in results_dir and plot their efficiency curves with distinct colors and legend,
    using inverted x-axis.

    Args:
        results_dir (Path or str): Directory containing CSV files.
    """
    results_dir = Path(results_dir)
    csv_files = [f for f in results_dir.iterdir() if f.suffix == '.csv']

    plt.figure(figsize=(10, 6))

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        filename = csv_file.name

        try:
            num_nodes_str = filename.split("_nodes")[-1].replace(".csv", "")
            num_nodes = int(num_nodes_str)
        except (IndexError, ValueError):
            print(f"Warning: Could not extract num_nodes from filename '{filename}', skipping this file.")
            continue

        efficiencies = df['normalized_efficiency'].tolist()
        if efficiencies[0] == 1.0:
            efficiencies = efficiencies[1:]

        total_nodes = num_nodes
        num_removed = list(range(len(efficiencies) + 1))
        percent_remaining = [100 * (total_nodes - n) / total_nodes for n in num_removed]

        efficiencies = [1.0] + efficiencies

        plt.plot(percent_remaining, efficiencies, label=filename)

    plt.xlabel("Percentage of Nodes Remaining")
    plt.ylabel("Normalized Efficiency")
    plt.title("Efficiency Degradation Across Multiple Runs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gca().invert_xaxis()
    plt.show()


def plot_average_efficiency_with_area(results_dir):
    """
    Load all CSV files in results_dir, compute average efficiency curve,
    plot the average (red) line with red circles at data points and shade the upper area.
    Also compute and print statistics over individual curve areas.

    Args:
        results_dir (Path or str): Directory containing CSV files with 'normalized_efficiency' column.
                                   Other file types (e.g. .log) will be ignored.

    Returns:
        float: Area above the average efficiency curve
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

            # Extract number of nodes from filename
            filename = csv_file.name
            num_nodes_str = filename.split("_nodes")[-1].replace(".csv", "")
            total_nodes = int(num_nodes_str)
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

    # Find the minimum length of the runs to align them
    min_len = min(len(e) for e in all_efficiencies)
    truncated_efficiencies = [e[:min_len] for e in all_efficiencies]
    truncated_percent_remaining = all_percent_remaining[0][:min_len]

    # Compute average efficiency across runs
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
    area_above_avg = integrate.trapezoid(gap_above_avg, dx=(truncated_percent_remaining[0] - truncated_percent_remaining[-1]) / (len(gap_above_avg) - 1))

    print(f"Area above average efficiency line: {area_above_avg:.4f}\n")

    # Print individual areas and summary statistics
    df_areas = pd.DataFrame({'File': [f.name for f in csv_files], 'Area Above Curve': individual_areas})
    print(df_areas.to_string(index=False))

    print("\nStatistics over area above efficiency curves:")
    print(f"Mean   : {np.mean(individual_areas):.4f}")
    print(f"Median : {np.median(individual_areas):.4f}")
    print(f"Min    : {np.min(individual_areas):.4f}")
    print(f"Max    : {np.max(individual_areas):.4f}")
    print(f"Std Dev: {np.std(individual_areas):.4f}")

    return area_above_avg, df_areas