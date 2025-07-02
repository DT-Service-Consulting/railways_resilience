import os
import pickle

def load_all_subgraphs(base_dir="../pkl", max_per_type=2):
    """
    Loads pickled subgraphs organized in subfolders named by number of nodes.

    Parameters:
        base_dir (str): Base directory containing subfolders of graphs.
        max_per_type (int or None): Max graphs to load per num_node type. 
                                    If None, loads all available.

    Returns:
        dict: {num_nodes: [graph1, graph2, ...]}
    """
    subgraphs_by_size = {}

    for size_folder in os.listdir(base_dir):
        size_path = os.path.join(base_dir, size_folder)
        if os.path.isdir(size_path) and size_folder.isdigit():
            size = int(size_folder)
            subgraphs = []

            pkl_files = sorted([
                f for f in os.listdir(size_path) 
                if f.endswith(".pkl") and f.startswith("graph_")
            ])

            if max_per_type is not None:
                pkl_files = pkl_files[:max_per_type]

            for filename in pkl_files:
                file_path = os.path.join(size_path, filename)
                with open(file_path, "rb") as f:
                    graph = pickle.load(f)
                    subgraphs.append(graph)

            subgraphs_by_size[size] = subgraphs

    return subgraphs_by_size