# Public Transport Network Graph Modelling using GTFS Data

This repository provides tools for processing [GTFS](https://gtfs.org/) (General Transit Feed Specification) data to model public transport railway networks as graphs using **NetworkX**.

The project focuses on railway systems in:
- Belgium
- Ireland
- Netherlands (For future work)

The repository includes workflows for:
- GTFS preprocessing and cleaning
- Network generation
- Topological analysis
- Graph robustness experiments
- Efficiency and connectivity analysis
- Visualization of experimental results

The project also contains optimization experiments for improving the performance of large-scale graph computations.

---

# 📦 Repository Structure

```bash
railways_resilience/
├── data/
│   ├── belgium/                # Extracted GTFS text files for Belgium
│   ├── ireland/                # Extracted GTFS text files for Ireland
│   ├── pkl/                    # Serialized NetworkX graphs and subnetworks
│   ├── results/                # Experiment results and processed outputs
│   ├── sqlite/                 # SQLite GTFS databases
│   └── zip/                    # Original GTFS zip files
│
├── graphs/                     # Network and subnetwork visualizations
│
├── notebooks/
│   ├── functions/
│   │   └── plot.py             # Functions for plotting NetworkX graphs and experiment result plots
│   │
│   ├── Marco/
│   │   ├── illustration.ipynb                  # Generates frames during node/edge removals and creates GIF visualizations
│   │   ├── Comparision of Curves.ipynb         # Visualization and comparison of experiment result curves
│   │   └── fitting_patterns/                   # Curve fitting experiments using Exponential, Logistic, Beta-like and related functions
│   │
│   ├── Data Cleaning/
│   │   ├── GTFS Railways.ipynb                 # Converts GTFS zip files into NetworkX railway graphs and performs preliminary analysis
│   │   ├── CheckNodes&Routes.ipynb             # Detailed analysis and cleaning of nodes and edges in the railway graph
│   │   └── OutlierEdgesRemoval.ipynb           # Removes edges with low train frequency from the network
│   │
│   └── Data Analysis/
│       ├── Analyzing Subgraphs.ipynb                   # Analysis of Belgian subnetworks based on topological properties
│       ├── Changes in Topological Indicators.ipynb     # Tracks changes in topological indicators during experiments
│       ├── Comparing Optimized Functions.ipynb         # Comparison of optimized implementations of core functions
│       ├── Efficiency (Batched Subgraphs).ipynb        # Efficiency analysis on batches of subnetworks
│       ├── Efficiency Of Networks (Full Graphs).ipynb  # Efficiency analysis on complete railway graphs
│       ├── Efficiency Of Subgraph-150 Nodes.ipynb      # Efficiency experiments on subnetworks of 150 nodes
│       ├── GTC_Belgium.ipynb                           # Graph Topological Characteristics analysis for Belgium
│       ├── GTC_Ireland.ipynb                           # Graph Topological Characteristics analysis for Ireland
│       ├── GTC_Netherlands.ipynb                       # Graph Topological Characteristics analysis for Netherlands
│       ├── Normalized-to-Actual.ipynb                  # Converts normalized experimental values back to actual values
│       ├── numConnected.ipynb                          # Tracks the number of connected components during experiments
│       ├── Results.ipynb                               # Visualization and aggregation of experiment results
│       ├── TopologicalIndicators_Belgium.ipynb         # Topological indicator analysis for Belgian railways
│       ├── TopologicalIndicators_Ireland.ipynb         # Topological indicator analysis for Irish railways
│       └── TopologicalIndicators_Netherlands.ipynb     # Topological indicator analysis for Dutch railways
│
├── scripts/                    # Python workflows for executing various removal strategy experiments
│
├── utils/                      # Import files for functions from the gtfs_railways package
│
├── config.py
├── INSTALL.md
├── LICENSE
├── NOTICE
└── README.md
```

---


# 🗄️ Dataset Downloads

Some GTFS datasets and SQLite files are not included in the repository due to their size.

Download the required datasets from:

[Download SQLite Files](https://www.dropbox.com/scl/fi/hd4l1vxb43j10tglrl4x5/sqlite.zip?rlkey=htpb057n5ibygd0p1iyldn42z&st=2xrzsyo5&dl=0)

After downloading, place the files inside the `data/` directory following this structure:

```bash
data/
├── belgium/
├── ireland/
├── pkl/
├── results/
├── sqlite/
└── zip/
```

Example:

```bash
data/sqlite/
├── belgium.sqlite
├── ireland.sqlite
└── netherlands.sqlite
```

```bash
data/zip/
├── belgium_gtfs.zip
├── ireland_gtfs.zip
└── netherlands_gtfs.zip
```

---


# 🧹 Data Cleaning Workflow

The preprocessing and cleaning pipeline consists of several stages:

1. **GTFS Parsing**
   - Conversion of GTFS zip files into NetworkX railway graphs.

2. **Node and Route Analysis**
   - Identification and removal of invalid or unnecessary nodes and edges.

3. **Outlier Edge Removal**
   - Removal of edges with low train frequency.

4. **Subnetwork Generation**
   - Creation and analysis of subnetworks for robustness experiments.

5. **Graph Serialization**
   - Saving processed graphs as `.pkl` files for later analysis.

---

# 📊 Data Analysis and Experiments

The repository contains several notebooks for network analysis and robustness experiments, including:

- Graph Topological Characteristics (GTC)
- Efficiency analysis on full networks and subnetworks
- Connected component analysis
- Topological indicator tracking during node removals
- Normalized-to-actual value conversions
- Performance optimization benchmarking
- Experimental result visualization

Additional notebooks are provided for:
- Curve fitting of degradation patterns
- Comparative visualization of different removal strategies
- Animated illustrations of node/edge removal experiments

---

# ⚡ Optimization

Multiple optimized implementations of the core functions are included to improve performance for large railway graphs and repeated experiment runs.

Core functions include:

```python
get_all_GTC()
```

```python
P_space()
```

Additional optimization benchmarking can be found in:

```bash
notebooks/Data Analysis/Comparing Optimized Functions.ipynb
```

---

# 🗄️ GTFS and SQLite Files

The repository supports GTFS datasets in:
- ZIP format
- Extracted TXT format
- SQLite databases

Place the datasets inside the `data/` directory following the structure shown above.

---

# 📈 Graph Visualizations

The `graphs/` directory contains:
- Railway network visualizations
- Cleaned subnetworks
- Experimental graph states
- Robustness analysis figures

---

# 🚀 Running Experiments

The `scripts/` directory contains Python workflows for executing different node and edge removal strategies used in the robustness experiments.

The notebooks in `Data Analysis/` can then be used to:
- Evaluate results
- Generate plots
- Compare strategies
- Analyze topological changes

---

# 📚 Dependencies

Please refer to:

```bash
INSTALL.md
```

for installation instructions and required dependencies.

---

# 📝 Notes

- The project primarily uses **NetworkX** for graph modelling and analysis.
- Experimental outputs and intermediate graphs are stored as `.pkl` files.
- Large GTFS datasets and generated results are intentionally excluded from version control where necessary.
