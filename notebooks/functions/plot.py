import matplotlib.pyplot as plt # type: ignore
import matplotlib.cm as cm # type: ignore
from matplotlib.lines import Line2D # type: ignore
import numpy as np 
import pandas as pd # type: ignore
from scipy import integrate # type: ignore
from pathlib import Path
import re
import textwrap
import ast

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

def plot_nodes_highlight(G, nodes, back_map="OSM", MAPS_API_KEY=None, name_attr="name"):
    """
    Plot the graph with given nodes highlighted in different colors and
    print the highlighted node names after the plot.

    Args:
        G (nx.Graph): Graph with 'lat' and 'lon' node attributes.
        nodes (list): Node IDs to highlight.
        back_map (str): "OSM", "GMAPS", or None.
        MAPS_API_KEY (str, optional): Required if back_map == "GMAPS".
        name_attr (str): Node attribute to use as the node name.
    """
    if not isinstance(nodes, (list, tuple, set)):
        nodes = [nodes]

    if back_map == "GMAPS":
        first_node = next(iter(G.nodes(data=True)))
        map_options = GMapOptions(
            lat=first_node[1]["lat"],
            lng=first_node[1]["lon"],
            map_type="roadmap",
            zoom=11
        )
        p = gmap(MAPS_API_KEY, map_options)
    else:
        p = figure(
            height=600,
            width=950,
            toolbar_location="below",
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

    # Build node position dictionary
    pos_dict = {}
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)

    for node_id, d in G.nodes(data=True):
        if back_map == "OSM":
            x, y = transformer.transform(float(d["lon"]), float(d["lat"]))
        else:
            x, y = float(d["lon"]), float(d["lat"])
        pos_dict[node_id] = (x, y)

    graph = from_networkx(G, layout_function=pos_dict)

    # Default node/edge styling
    graph.node_renderer.glyph = Circle(size=7, fill_color="gray")
    graph.edge_renderer.glyph = MultiLine(line_width=2, line_alpha=0.5)

    # Choose color palette
    palette = Category20[20] if len(nodes) > 10 else Category10[10]

    printed_nodes = []

    # Highlight each requested node
    for idx, node in enumerate(nodes):
        if node in pos_dict:
            color = palette[idx % len(palette)]
            x, y = [pos_dict[node][0]], [pos_dict[node][1]]
            p.circle(x=x, y=y, size=15, color=color, legend_label=f"Node {node}")

            node_name = G.nodes[node].get(name_attr, "Unknown")
            printed_nodes.append(f"({node_name})({node})")
        else:
            print(f"Node {node} not found in graph")

    # Add hover tools
    node_hover_tool = HoverTool(
        tooltips=[("index", "@index"), ("name", "@name")],
        renderers=[graph.node_renderer]
    )
    p.add_tools(node_hover_tool)

    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    p.renderers.append(graph)

    if back_map == "OSM":
        p.add_tile(get_provider(Vendors.CARTODBPOSITRON))

    if printed_nodes:
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

    show(p)

    print("\nHighlighted nodes:")
    for node_str in printed_nodes:
        print(node_str)

def plot_edges_highlight(G, edges, back_map="OSM", MAPS_API_KEY=None, name_attr="name"):
    """
    Plot the graph with given edges highlighted in different colors and
    print edge names after the plot.

    Args:
        G (nx.Graph): Graph with 'lat' and 'lon' node attributes.
        edges (list): List of edge tuples (u, v) to highlight.
        back_map (str): "OSM", "GMAPS", or None.
        MAPS_API_KEY (str, optional): Required if back_map == "GMAPS".
        name_attr (str): Node attribute to use as the node name.
    """

    if not isinstance(edges, (list, tuple, set)):
        edges = [edges]

    if back_map == "GMAPS":
        first_node = next(iter(G.nodes(data=True)))
        map_options = GMapOptions(
            lat=first_node[1]["lat"],
            lng=first_node[1]["lon"],
            map_type="roadmap",
            zoom=11
        )
        p = gmap(MAPS_API_KEY, map_options)
    else:
        p = figure(
            height=600,
            width=950,
            toolbar_location="below",
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

    pos_dict = {}
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)

    for node_id, d in G.nodes(data=True):
        if back_map == "OSM":
            x, y = transformer.transform(float(d["lon"]), float(d["lat"]))
        else:
            x, y = float(d["lon"]), float(d["lat"])
        pos_dict[node_id] = (x, y)

    graph = from_networkx(G, layout_function=pos_dict)
    graph.node_renderer.glyph = Circle(size=6, fill_color="gray")
    graph.edge_renderer.glyph = MultiLine(line_width=2, line_alpha=0.3)

    p.renderers.append(graph)

    palette = Category20[20] if len(edges) > 10 else Category10[10]
    printed_edges = []

    for idx, edge in enumerate(edges):
        if len(edge) < 2:
            print(f"Invalid edge format: {edge}")
            continue

        u, v = edge[0], edge[1]

        # For undirected graphs, accept reversed order too
        edge_exists = G.has_edge(u, v) or (not G.is_directed() and G.has_edge(v, u))

        if u in pos_dict and v in pos_dict and edge_exists:
            color = palette[idx % len(palette)]

            x = [pos_dict[u][0], pos_dict[v][0]]
            y = [pos_dict[u][1], pos_dict[v][1]]

            p.line(
                x, y,
                line_width=5,
                color=color,
                alpha=0.9,
                legend_label=f"{u}-{v}"
            )

            u_name = G.nodes[u].get(name_attr, "Unknown")
            v_name = G.nodes[v].get(name_attr, "Unknown")
            printed_edges.append(f"({u_name})({u}) to ({v_name})({v})")
        else:
            print(f"Edge ({u}, {v}) not found in graph")

    node_hover_tool = HoverTool(
        tooltips=[("index", "@index"), ("name", "@name")],
        renderers=[graph.node_renderer]
    )
    p.add_tools(node_hover_tool)

    p.toolbar.active_scroll = p.select_one(WheelZoomTool)

    if back_map == "OSM":
        p.add_tile(get_provider(Vendors.CARTODBPOSITRON))

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    show(p)

    print("\nHighlighted edges:")
    for edge_str in printed_edges:
        print(edge_str)

def plot_full_graph_with_highlighted_edges(G, edges_dict, back_map="OSM", MAPS_API_KEY=None):
    """
    Plots the entire L_graph as base (all edges + all nodes),
    then overlays selected OD pairs in different colors depending on n_vehicles value.
    
    Args:
        G (nx.Graph): Graph with lat/lon attributes.
        edges_dict (dict): {n_veh_value: [(u, v, data), ...]}
    """

    # ----------------------
    # Base map setup
    # ----------------------
    if back_map == "GMAPS":
        first_node = next(iter(G.nodes(data=True)))
        map_options = GMapOptions(
            lat=float(first_node[1]["lat"]),
            lng=float(first_node[1]["lon"]),
            map_type="roadmap",
            zoom=11,
        )
        p = gmap(MAPS_API_KEY, map_options)
    else:
        p = figure(
            height=600, width=950,
            toolbar_location="below",
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

    # ----------------------
    # Compute projected positions
    # ----------------------
    pos_dict = {}
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)

    for node, d in G.nodes(data=True):
        if back_map == "OSM":
            x, y = transformer.transform(float(d["lon"]), float(d["lat"]))
        else:
            x, y = float(d["lon"]), float(d["lat"])
        pos_dict[node] = (x, y)

    # ----------------------
    # Render entire L_graph as base (ALL edges, ALL nodes)
    # ----------------------
    base_graph = from_networkx(G, layout_function=pos_dict)

    # Nodes style
    base_graph.node_renderer.glyph = Circle(size=5, fill_color="gray", fill_alpha=0.5)

    # Edges style
    base_graph.edge_renderer.glyph = MultiLine(line_alpha=0.4, line_width=1.5, line_color="gray")

    # Add base graph
    p.renderers.append(base_graph)

    # ----------------------
    # Palette for highlighting
    # ----------------------
    palette = Category20[20] if len(edges_dict) > 10 else Category10[10]

    # ----------------------
    # Overlay highlighted OD edges
    # ----------------------
    for idx, (n_val, edges) in enumerate(edges_dict.items()):
        color = palette[idx % len(palette)]
        label = f"{n_val} vehicles"

        for u, v, data in edges:
            if u not in pos_dict or v not in pos_dict:
                continue

            x0, y0 = pos_dict[u]
            x1, y1 = pos_dict[v]

            # Highlight edge
            p.line([x0, x1], [y0, y1],
                   line_width=4, alpha=0.9, color=color,
                   legend_label=label)

            # Highlight nodes
            p.circle([x0, x1], [y0, y1],
                     size=10, alpha=1.0, color=color)

    # ----------------------
    # Hover + tiles
    # ----------------------
    hover = HoverTool(tooltips=[("Node", "@index")])
    p.add_tools(hover)

    if back_map == "OSM":
        p.add_tile(get_provider(Vendors.CARTODBPOSITRON))

    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
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


def plot_efficiency_comparison_single(
    run_configs,
    title="",
    xlim=None,
    save_path=None
):
    """
    Plot efficiency curves for multiple runs on the same graph,
    shading the area above each curve and calculating two area values:
    1. Area above the full curve over the entire 0-100% range
    2. Area above the visible curve within the plotted x-range

    Args:
        run_configs (list of dict): Each dict must have keys:
            'fil' (Path or str to CSV file), 'color', 'label'
        title (str): Plot title
        xlim (tuple, optional): (min_x, max_x) range for zoom.
        save_path (Path or str, optional): If provided, saves the figure.

    Returns:
        tuple:
            areas_above_full (dict): area above each full curve
            areas_above_plot (dict): area above each curve within plotted x-range
    """

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 22,
        "axes.titlesize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })

    fig, ax = plt.subplots(figsize=(10, 9))

    areas_above_full = {}
    areas_above_plot = {}
    all_x_values = []
    legend_handles = []

    filename0 = Path(run_configs[0]["fil"]).name

    if "_nodes" in filename0:
        keyword = "_nodes"
    elif "_edges" in filename0:
        keyword = "_edges"
    else:
        raise ValueError("Filenames must contain either '_nodes' or '_edges'")

    for cfg in run_configs:
        df = pd.read_csv(cfg["fil"])
        efficiencies = df["normalized_efficiency"].tolist()

        if efficiencies[0] == 1.0:
            efficiencies = efficiencies[1:]

        efficiencies = [1.0] + efficiencies

        filename = Path(cfg["fil"]).name

        if keyword not in filename:
            raise ValueError(
                f"Inconsistent file naming: expected {keyword} in {filename}"
            )

        total_count = int(filename.split(keyword)[-1].replace(".csv", ""))

        num_removed = list(range(len(efficiencies)))

        percent_remaining = [
            100 * (total_count - n) / total_count
            for n in num_removed
        ]

        all_x_values.extend(percent_remaining)

        gap_above = [1 - x for x in efficiencies]

        area_full = integrate.trapezoid(
            gap_above,
            dx=100 / total_count
        )

        areas_above_full[cfg["label"]] = area_full

        if xlim is not None:
            xmin, xmax = min(xlim), max(xlim)

            zoom_x = []
            zoom_gap = []

            for x, g in zip(percent_remaining, gap_above):
                if xmin <= x <= xmax:
                    zoom_x.append(x)
                    zoom_gap.append(g)

            if len(zoom_x) > 1:
                area_plot = abs(integrate.trapezoid(zoom_gap, zoom_x))
            else:
                area_plot = 0.0
        else:
            area_plot = area_full

        areas_above_plot[cfg["label"]] = area_plot

        marker_style = "o"

        if cfg["label"].lower() == "netherlands":
            marker_style = "^"

        ax.plot(
            percent_remaining,
            efficiencies,
            color=cfg["color"],
            linewidth=2.5,
            label=cfg["label"]
        )

        ax.scatter(
            percent_remaining,
            efficiencies,
            color=cfg["color"],
            marker=marker_style,
            s=70,
            alpha=0.9
        )

        ax.fill_between(
            percent_remaining,
            efficiencies,
            1.0,
            color=cfg["color"],
            alpha=0.3
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=cfg["color"],
                marker=marker_style,
                linewidth=2.5,
                markersize=10,
                label=cfg["label"]
            )
        )

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(max(all_x_values), min(all_x_values))

    if ax.get_xlim()[0] < ax.get_xlim()[1]:
        ax.invert_xaxis()

    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks(np.linspace(0, 1, 6))

    ax.set_xlabel(
        f"Percentage of {keyword.strip('_').capitalize()} Remaining",
        fontsize=22,
        labelpad=14
    )

    ax.set_ylabel(
        "Normalized Efficiency",
        fontsize=22,
        labelpad=14
    )

    if title:
        ax.set_title(
            title,
            fontsize=24,
            pad=18
        )

    ax.grid(True, alpha=0.3)

    ax.legend(
        handles=legend_handles,
        fontsize=18,
        frameon=True
    )

    plt.subplots_adjust(
        left=0.12,
        right=1.0,
        bottom=0.12,
        top=0.92
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    print("\nArea above curve (full graph):")
    for label, area in areas_above_full.items():
        print(f"{label}: {area:.4f}")

    print("\nArea above curve (plot window):")
    for label, area in areas_above_plot.items():
        print(f"{label}: {area:.4f}")

    return areas_above_full, areas_above_plot


def plot_efficiency_comparison_multi(
    run_configs,
    title="Efficiency Comparison",
    xlim=None,
    save_path_left=None,
    save_path_right=None,
    show=True,
    show_titles=True
):
    """
    Plots individual efficiency curves and mean efficiency curves.

    Updated for publication:
    - Larger fonts
    - Thicker lines
    - Line styles for grayscale readability
    - Solid line for Belgium, dashed/dotted line for Netherlands via run_configs
    """

    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18
    })

    def load_curve_from_csv(csv_file):
        filename = csv_file.name

        if "_nodes" in filename:
            total_elements = int(filename.split("_nodes")[-1].replace(".csv", ""))
        elif "_edges" in filename:
            total_elements = int(filename.split("_edges")[-1].replace(".csv", ""))
        else:
            return None, None

        df = pd.read_csv(csv_file)
        if "normalized_efficiency" not in df.columns:
            return None, None

        efficiencies = df["normalized_efficiency"].tolist()

        if len(efficiencies) == 0:
            return None, None

        if efficiencies[0] == 1.0:
            efficiencies = efficiencies[1:]

        efficiencies = [1.0] + efficiencies

        num_removed = list(range(len(efficiencies)))
        percent_remaining = [
            100 * (total_elements - n) / total_elements
            for n in num_removed
        ]

        return percent_remaining, efficiencies

    def compute_area_above_curve(percent_remaining, efficiencies, xlim=None):
        gap_above = [1 - e for e in efficiencies]

        area_full = abs(integrate.trapezoid(gap_above, percent_remaining))

        if xlim is not None:
            xmin, xmax = min(xlim), max(xlim)

            zoom_x = []
            zoom_gap = []

            for x, g in zip(percent_remaining, gap_above):
                if xmin <= x <= xmax:
                    zoom_x.append(x)
                    zoom_gap.append(g)

            if len(zoom_x) > 1:
                area_plot = abs(integrate.trapezoid(zoom_gap, zoom_x))
            else:
                area_plot = 0.0
        else:
            area_plot = area_full

        return area_full, area_plot

    avg_areas_full = {}
    avg_areas_plot = {}

    # =========================
    # LEFT FIGURE
    # =========================
    fig_left, ax1 = plt.subplots(figsize=(8, 6))
    legend_elements = []
    plotted_left = False

    for config in run_configs:
        directory = Path(config["dir"])
        color = config.get("color", "black")
        label = config["label"]
        linestyle = config.get("linestyle", "-")

        csv_files = sorted([f for f in directory.iterdir() if f.suffix == ".csv"])

        for csv_file in csv_files:
            try:
                percent_remaining, efficiencies = load_curve_from_csv(csv_file)
                if percent_remaining is None:
                    continue

                ax1.plot(
                    percent_remaining,
                    efficiencies,
                    color=color,
                    linestyle=linestyle,
                    alpha=0.8,
                    linewidth=2
                )
                plotted_left = True

            except Exception:
                continue

        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2.5,
                linestyle=linestyle,
                label=label
            )
        )

    ax1.set_xlabel("Percentage Remaining", fontsize=20)
    ax1.set_ylabel("Normalized Efficiency", fontsize=20)
    ax1.tick_params(axis="both", labelsize=18)

    if show_titles:
        ax1.set_title(f"{title} - Individual Efficiency Curves", fontsize=22)

    ax1.grid(True)
    ax1.invert_xaxis()

    if xlim:
        ax1.set_xlim(xlim)

    if plotted_left:
        ax1.legend(handles=legend_elements, fontsize=18)

    fig_left.tight_layout()

    # =========================
    # RIGHT FIGURE
    # =========================
    fig_right, ax2 = plt.subplots(figsize=(7, 5))
    plotted_right = False

    for config in run_configs:
        directory = Path(config["dir"])
        color = config.get("color", "black")
        label = config["label"]
        linestyle = config.get("linestyle", "-")

        csv_files = sorted([f for f in directory.iterdir() if f.suffix == ".csv"])

        all_efficiencies = []
        all_percent_remaining = []
        area_full_runs = []
        area_plot_runs = []

        for csv_file in csv_files:
            try:
                percent_remaining, efficiencies = load_curve_from_csv(csv_file)
                if percent_remaining is None:
                    continue

                all_efficiencies.append(efficiencies)
                all_percent_remaining.append(percent_remaining)

                area_full, area_plot = compute_area_above_curve(
                    percent_remaining,
                    efficiencies,
                    xlim=xlim
                )

                area_full_runs.append(area_full)
                area_plot_runs.append(area_plot)

            except Exception:
                continue

        if not all_efficiencies:
            continue

        avg_areas_full[label] = float(np.mean(area_full_runs)) if area_full_runs else 0.0
        avg_areas_plot[label] = float(np.mean(area_plot_runs)) if area_plot_runs else 0.0

        min_len = min(len(e) for e in all_efficiencies)
        eff_matrix = np.array([e[:min_len] for e in all_efficiencies])
        pr = np.array(all_percent_remaining[0][:min_len])

        mean_eff = eff_matrix.mean(axis=0)
        std_eff = eff_matrix.std(axis=0)

        ax2.plot(
            pr,
            mean_eff,
            color=color,
            linestyle=linestyle,
            linewidth=3,
            label=label
        )

        ax2.fill_between(
            pr,
            mean_eff - std_eff,
            mean_eff + std_eff,
            color=color,
            alpha=0.18
        )

        plotted_right = True

    ax2.set_xlabel("Percentage Remaining", fontsize=20)
    ax2.set_ylabel("Normalized Efficiency", fontsize=20)
    ax2.tick_params(axis="both", labelsize=18)

    if show_titles:
        ax2.set_title(f"{title} - Average Efficiency with Std Deviation", fontsize=22)

    ax2.grid(True)
    ax2.invert_xaxis()

    if xlim:
        ax2.set_xlim(xlim)

    if plotted_right:
        ax2.legend(fontsize=18)

    fig_right.tight_layout()

    # =========================
    # SAVE
    # =========================
    if save_path_left is not None:
        save_path_left = Path(save_path_left)
        save_path_left.parent.mkdir(parents=True, exist_ok=True)
        fig_left.savefig(save_path_left, dpi=300, bbox_inches="tight")

    if save_path_right is not None:
        save_path_right = Path(save_path_right)
        save_path_right.parent.mkdir(parents=True, exist_ok=True)
        fig_right.savefig(save_path_right, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig_left)
        plt.close(fig_right)

    print("\nAverage area above curve (full graph):")
    for label, area in avg_areas_full.items():
        print(f"{label}: {area:.4f}")

    print("\nAverage area above curve (plot window):")
    for label, area in avg_areas_plot.items():
        print(f"{label}: {area:.4f}")

    return avg_areas_full, avg_areas_plot


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

import textwrap

def plot_efficiency_with_node_labels_overlay(
    df1,
    df2,
    label1="Run 1",
    label2="Run 2",
    color1="black",
    color2="orange",
    marker1="o",
    marker2="^",
    title=None,
    save_path=None
):
    """
    Overlay two efficiency–node-removal plots using twin x-axes
    with publication-quality visuals.
    """

    # =========================
    # Helper for multiline labels
    # =========================
    def wrap_labels(labels, width=12):
        wrapped = []
        for label in labels:
            wrapped.append("\n".join(textwrap.wrap(label, width=width)))
        return wrapped

    # =========================
    # Extract data
    # =========================
    eff1 = df1["normalized_efficiency"].tolist()
    nodes1 = df1["removed_node_names"].tolist()

    eff2 = df2["normalized_efficiency"].tolist()
    nodes2 = df2["removed_node_names"].tolist()

    # Prepend full graph
    if eff1[0] != 1.0:
        eff1 = [1.0] + eff1
        nodes1 = [""] + nodes1

    if eff2[0] != 1.0:
        eff2 = [1.0] + eff2
        nodes2 = [""] + nodes2

    x1 = list(range(len(eff1)))
    x2 = list(range(len(eff2)))

    labels1 = ["Full Graph"] + nodes1[1:]
    labels2 = ["Full Graph"] + nodes2[1:]

    # Wrap labels into multiple lines
    labels1 = wrap_labels(labels1, width=12)
    labels2 = wrap_labels(labels2, width=12)

    # =========================
    # Publication-quality fonts
    # =========================
    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 26,
        "axes.labelsize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 18,
        "legend.fontsize": 20
    })

    # =========================
    # Plot
    # =========================
    fig, ax1 = plt.subplots(figsize=(18, 9))
    ax2 = ax1.twiny()

    ax1.set_ylim(0, 1)

    # Belgium
    ax1.plot(
        x1,
        eff1,
        color=color1,
        marker=marker1,
        linewidth=2.8,
        markersize=7,
        label=label1
    )

    ax1.fill_between(
        x1,
        eff1,
        1.0,
        color=color1,
        alpha=0.25
    )

    # Netherlands
    ax2.plot(
        x2,
        eff2,
        color=color2,
        marker=marker2,
        linewidth=2.8,
        markersize=8,
        label=label2
    )

    ax2.fill_between(
        x2,
        eff2,
        1.0,
        color=color2,
        alpha=0.25
    )

    # =========================
    # Bottom x-axis (Belgium)
    # =========================
    ax1.set_xticks(x1)

    ax1.set_xticklabels(
        labels1,
        rotation=0,
        ha="center",
        linespacing=1.2
    )

    ax1.set_xlabel(
        f"Removed Nodes ({label1})",
        fontsize=22,
        labelpad=4
    )

    # =========================
    # Top x-axis (Netherlands)
    # =========================
    ax2.set_xticks(x2)

    ax2.set_xticklabels(
        labels2,
        rotation=0,
        ha="center",
        linespacing=1.2
    )

    ax2.set_xlabel(
        f"Removed Nodes ({label2})",
        fontsize=22,
        labelpad=20   # extra spacing from labels
    )

    # =========================
    # Y-axis
    # =========================
    ax1.set_ylabel(
        "Normalized Efficiency",
        fontsize=22
    )

    # =========================
    # Tick spacing
    # =========================
    ax1.tick_params(
        axis='x',
        labelsize=16,
        pad=10
    )

    ax2.tick_params(
        axis='x',
        labelsize=16,
        pad=10
    )

    ax1.tick_params(
        axis='y',
        labelsize=18
    )

    # =========================
    # Title
    # =========================
    if title:
        ax1.set_title(title, fontsize=26)

    # =========================
    # Grid
    # =========================
    ax1.grid(True, alpha=0.4)

    # =========================
    # Legend
    # =========================
    lines1, labels1_ = ax1.get_legend_handles_labels()
    lines2, labels2_ = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1_ + labels2_,
        loc="best",
        fontsize=20
    )

    # More room for top/bottom labels
    plt.subplots_adjust(
        top=0.82,
        bottom=0.25
    )

    # =========================
    # Save
    # =========================
    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    # =========================
    # Areas
    # =========================
    area1 = integrate.trapezoid([1 - x for x in eff1], dx=1)
    area2 = integrate.trapezoid([1 - x for x in eff2], dx=1)

    print(f"Area above curve ({label1}): {area1:.4f}")
    print(f"Area above curve ({label2}): {area2:.4f}")

    return area1, area2

def plot_efficiency_comparison_with_named_inset(
    run_configs,
    title="",
    xlim=None,
    save_path=None,
    inset_graphs=None,
    inset_n=5,
    inset_panel_position=(0.35, 0.30, 0.63, 0.69),
    inset_plot_rect=(0.08, 0.25, 0.88, 0.50),
    inset_title="First Removed Elements",
    main_legend_loc="lower left",
    inset_legend=False,
    wrap_width=14,
):

    fig, ax = plt.subplots(figsize=(10, 6))
    loaded_dfs = []

    filename0 = Path(run_configs[0]["fil"]).name
    if "_nodes" in filename0:
        keyword = "_nodes"
        removal_kind = "node"
    elif "_edges" in filename0:
        keyword = "_edges"
        removal_kind = "edge"
    else:
        raise ValueError("Could not infer whether this is a node or edge removal file.")

    # -----------------------------
    # Helpers
    # -----------------------------
    def parse_node_id(val):
        if pd.isna(val) or val == "":
            return None
        try:
            return int(float(val))
        except Exception:
            return None

    def parse_edge(val):
        if pd.isna(val) or val == "":
            return None
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, tuple) and len(parsed) == 2:
                u, v = parsed
                return int(u), int(v)
        except Exception:
            pass
        return None

    def node_name_lookup(graph, node_id):
        if node_id is not None and node_id in graph.nodes:
            return graph.nodes[node_id].get("name", str(node_id))
        return str(node_id)

    def edge_name_lookup(graph, edge_tuple):
        if edge_tuple is None:
            return ""

        u, v = edge_tuple
        u_name = node_name_lookup(graph, u)
        v_name = node_name_lookup(graph, v)

        if graph.has_edge(u, v):
            edge_data = graph.get_edge_data(u, v)
            if isinstance(edge_data, dict):
                edge_name = edge_data.get("name")
                if edge_name:
                    return str(edge_name)

        return f"{u_name} - {v_name}"

    def format_station_label(name, max_chars=14):
        if not name:
            return ""

        name = str(name).strip()

        if name.lower() == "full graph":
            return "Full\nGraph"

        return textwrap.fill(name, width=max_chars, break_long_words=False, break_on_hyphens=True)

    def get_removed_labels(df, graph, kind, n):
        raw_vals = df["removed_node"].tolist()[: n + 1]

        if kind == "node":
            parsed = [parse_node_id(x) for x in raw_vals]
            labels = ["Full Graph"] + [node_name_lookup(graph, x) for x in parsed[1:]]
        else:
            parsed = [parse_edge(x) for x in raw_vals]
            labels = ["Full Graph"] + [edge_name_lookup(graph, x) for x in parsed[1:]]

        return [format_station_label(x, max_chars=wrap_width) for x in labels]

    # -----------------------------
    # Main plot
    # -----------------------------
    for cfg in run_configs:
        df = pd.read_csv(cfg["fil"])
        loaded_dfs.append(df)

        efficiencies = df["normalized_efficiency"].tolist()
        if efficiencies and efficiencies[0] == 1.0:
            efficiencies = efficiencies[1:]
        efficiencies = [1.0] + efficiencies

        total_count = int(Path(cfg["fil"]).name.split(keyword)[-1].replace(".csv", ""))

        num_removed = list(range(len(efficiencies)))
        percent_remaining = [100 * (total_count - n) / total_count for n in num_removed]

        ax.plot(
            percent_remaining,
            efficiencies,
            color=cfg["color"],
            label=cfg["label"],
            linewidth=1.8,
        )
        ax.scatter(
            percent_remaining,
            efficiencies,
            color=cfg["color"],
            s=15,
            alpha=0.7,
        )
        ax.fill_between(
            percent_remaining,
            efficiencies,
            1.0,
            color=cfg["color"],
            alpha=0.3,
        )

    if xlim:
        ax.set_xlim(xlim)
    if ax.get_xlim()[0] < ax.get_xlim()[1]:
        ax.invert_xaxis()

    xlabel = "Percentage of Nodes Remaining" if removal_kind == "node" else "Percentage of Edges Remaining"

    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized Efficiency")
    ax.set_title(title)
    ax.tick_params(axis="both")
    ax.grid(True, alpha=0.6)
    ax.legend(loc=main_legend_loc)

    # -----------------------------
    # Inset
    # -----------------------------
    if inset_graphs is not None:
        df1, df2 = loaded_dfs
        g1, g2 = inset_graphs

        eff1 = df1["normalized_efficiency"].tolist()[: inset_n + 1]
        eff2 = df2["normalized_efficiency"].tolist()[: inset_n + 1]

        labels1 = get_removed_labels(df1, g1, removal_kind, inset_n)
        labels2 = get_removed_labels(df2, g2, removal_kind, inset_n)

        x1 = list(range(len(eff1)))
        x2 = list(range(len(eff2)))

        panel = ax.inset_axes(inset_panel_position)
        panel.set_facecolor("white")
        panel.set_xticks([])
        panel.set_yticks([])

        for spine in panel.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

        panel.text(0.5, 0.99, inset_title, ha="center", va="top", fontsize=9)

        if removal_kind == "node":
            panel.text(0.5, 0.93, "Removed Nodes (Netherlands)", ha="center", fontsize=8)
            panel.text(0.5, 0.05, "Removed Nodes (Belgium)", ha="center", fontsize=8)
        else:
            panel.text(0.5, 0.93, "Removed Edges (Netherlands)", ha="center", fontsize=8)
            panel.text(0.5, 0.03, "Removed Edges (Belgium)", ha="center", fontsize=8)

        panel.text(0.01, 0.5, "Normalized Efficiency", rotation=90, va="center", fontsize=8)

        axins = panel.inset_axes(inset_plot_rect)
        axins_top = axins.twiny()

        axins.set_ylim(0.0, 1.0)
        axins.set_yticks(np.linspace(0, 1, 6))
        axins_top.set_ylim(0.0, 1.0)

        axins.plot(x1, eff1, color="black", marker="o", markersize=4)
        axins_top.plot(x2, eff2, color="#FF5A00", marker="o", markersize=4)

        axins.set_xticks(x1)
        axins.set_xticklabels(labels1, fontsize=7, rotation=0, ha="center")

        axins_top.set_xticks(x2)
        axins_top.set_xticklabels(labels2, fontsize=7, rotation=0, ha="center")

        axins.tick_params(axis="x", pad=2)
        axins_top.tick_params(axis="x", pad=6)
        axins.tick_params(axis="y", labelsize=8)
        axins.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_inset_only(
    run_configs,
    inset_graphs,
    inset_n=5,
    title=None,
    save_path=None,
    wrap_width=10,
    figsize=(14, 8),
    show_xaxis_names=True,
):
    """
    Standalone inset-style comparison plot with:
    - no rotated labels
    - wrapped labels
    - increased spacing
    - publication-friendly sizing
    """

    filename0 = Path(run_configs[0]["fil"]).name

    if "_nodes" in filename0:
        removal_kind = "node"
    elif "_edges" in filename0:
        removal_kind = "edge"
    else:
        raise ValueError("Could not infer node/edge removal type.")

    def parse_node_id(val):
        if pd.isna(val) or val == "":
            return None

        try:
            return int(float(val))
        except Exception:
            return None

    def parse_edge(val):
        if pd.isna(val) or val == "":
            return None

        try:
            parsed = ast.literal_eval(str(val))

            if isinstance(parsed, tuple) and len(parsed) == 2:
                return int(parsed[0]), int(parsed[1])

        except Exception:
            pass

        return None

    def node_name_lookup(graph, node_id):
        if node_id is not None and node_id in graph.nodes:
            return graph.nodes[node_id].get("name", str(node_id))

        return str(node_id)

    def edge_name_lookup(graph, edge_tuple):
        if edge_tuple is None:
            return ""

        u, v = edge_tuple

        u_name = node_name_lookup(graph, u)
        v_name = node_name_lookup(graph, v)

        if graph.has_edge(u, v):
            edge_data = graph.get_edge_data(u, v)

            if isinstance(edge_data, dict):
                edge_name = edge_data.get("name")

                if edge_name:
                    return str(edge_name)

        return f"{u_name} - {v_name}"

    def format_label(name, max_chars=10):

        if not name:
            return ""

        name = str(name).strip()

        if name.lower() == "full graph":
            return "Full\nGraph"

        name = name.replace(" - ", "\n-\n")

        return textwrap.fill(
            name,
            width=max_chars,
            break_long_words=False,
            break_on_hyphens=True,
        )

    def get_removed_labels(df, graph, kind, n):

        raw_vals = df["removed_node"].tolist()[: n + 1]

        if kind == "node":

            parsed = [parse_node_id(x) for x in raw_vals]

            labels = ["Full Graph"] + [
                node_name_lookup(graph, x)
                for x in parsed[1:]
            ]

        else:

            parsed = [parse_edge(x) for x in raw_vals]

            labels = ["Full Graph"] + [
                edge_name_lookup(graph, x)
                for x in parsed[1:]
            ]

        return [
            format_label(x, max_chars=wrap_width)
            for x in labels
        ]

    dfs = [pd.read_csv(cfg["fil"]) for cfg in run_configs]

    df1, df2 = dfs
    g1, g2 = inset_graphs

    eff1 = df1["normalized_efficiency"].tolist()[: inset_n + 1]
    eff2 = df2["normalized_efficiency"].tolist()[: inset_n + 1]

    labels1 = get_removed_labels(
        df1,
        g1,
        removal_kind,
        inset_n
    )

    labels2 = get_removed_labels(
        df2,
        g2,
        removal_kind,
        inset_n
    )

    spacing = 1.6

    x1 = np.arange(len(eff1)) * spacing
    x2 = np.arange(len(eff2)) * spacing

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 22,
        "axes.titlesize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })

    fig, ax = plt.subplots(figsize=figsize)

    ax_top = ax.twiny()

    ax.set_ylim(0.0, 1.0)
    ax_top.set_ylim(0.0, 1.0)

    ax.plot(
        x1,
        eff1,
        color=run_configs[0]["color"],
        marker="o",
        linewidth=2.5,
        markersize=7,
        label=run_configs[0]["label"],
    )

    ax_top.plot(
        x2,
        eff2,
        color=run_configs[1]["color"],
        marker="^",
        linewidth=2.5,
        markersize=7,
        label=run_configs[1]["label"],
    )

    ax.set_xticks(x1)

    ax.set_xticklabels(
        labels1,
        rotation=0,
        ha="center",
        fontsize=18,
    )

    ax_top.set_xticks(x2)

    ax_top.set_xticklabels(
        labels2,
        rotation=0,
        ha="center",
        fontsize=18,
    )

    if show_xaxis_names:

        if removal_kind == "node":

            ax.set_xlabel(
                "Removed Nodes (Belgium)",
                fontsize=22,
                labelpad=20
            )

            ax_top.set_xlabel(
                "Removed Nodes (Netherlands)",
                fontsize=22,
                labelpad=20
            )

        else:

            ax.set_xlabel(
                "Removed Edges (Belgium)",
                fontsize=22,
                labelpad=20
            )

            ax_top.set_xlabel(
                "Removed Edges (Netherlands)",
                fontsize=22,
                labelpad=20
            )

    ax.set_ylabel(
        "Normalized Efficiency",
        fontsize=22,
        labelpad=16
    )

    if title is not None:
        ax.set_title(
            title,
            fontsize=26,
            pad=22
        )

    ax.tick_params(
        axis="x",
        pad=18
    )

    ax_top.tick_params(
        axis="x",
        pad=18
    )

    ax.tick_params(
        axis="y",
        labelsize=18
    )

    ax.grid(
        True,
        alpha=0.3
    )

    lines1, labels_1 = ax.get_legend_handles_labels()
    lines2, labels_2 = ax_top.get_legend_handles_labels()

    ax.legend(
        lines1 + lines2,
        labels_1 + labels_2,
        loc="lower left",
        fontsize=18,
        frameon=True,
    )

    plt.subplots_adjust(
        top=0.78,
        bottom=0.32,
        left=0.10,
        right=0.97
    )

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()