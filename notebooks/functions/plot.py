import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from bokeh.plotting import figure, show, from_networkx
from bokeh.models import Circle, MultiLine, HoverTool, LinearColorMapper, ColorBar, WheelZoomTool
from bokeh.tile_providers import get_provider, Vendors
from bokeh.io.export import export_png
from pyproj import Transformer
from bokeh.models import GMapOptions
from bokeh.plotting import gmap
import networkx as nx

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

def plot_efficiency_results(num_removed, efficiencies, title="Impact of Node Removal on Network Efficiency (Normalized)"):
    """
    Plots the change in normalized efficiency as nodes are removed.

    Parameters:
    - num_removed: List of number of nodes removed
    - efficiencies: Corresponding list of normalized efficiencies
    - title: Plot title
    """
    plt.figure(figsize=(6, 4))
    plt.plot(num_removed, efficiencies, marker='o')
    plt.xlabel("Number of Nodes Removed")
    plt.ylabel("Normalized Efficiency")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_efficiency_results_from_batch(row):
    """
    Plot the efficiency drop across node removals for a single subgraph.

    Parameters:
    row (pd.Series): A row from the DataFrame containing the following keys:
        - 'original_efficiency': efficiency before any node removal
        - 'efficiency_after_each_removal': list of efficiency values after each node is removed
        - 'num_nodes': number of nodes in the subgraph
        - 'graph_index': index of the subgraph within its group

    The function combines the original efficiency with the efficiency after each removal,
    and plots them as a line chart with points for visual tracking of efficiency drop.
    """
    # Full efficiency list: original + after each removal
    all_efficiencies = row['efficiency_after_each_removal']
    num_removed = list(range(1, len(all_efficiencies) + 1))
    plot_efficiency_results(num_removed, all_efficiencies)


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
    df_results["num_nodes_removed"] = df_results["removed_nodes"].apply(len)

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
        "Node Removed": row["removed_nodes"],
        "Time Elapsed (s)": individual_times
    })
    display(df)

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