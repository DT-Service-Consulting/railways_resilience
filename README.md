# Public Transport Network Graph Modelling using GTFS Data

This repository provides tools for processing [GTFS](https://github.com/DT-Service-Consulting/gtfs_railways/tree/main) (General Transit Feed Specification) data to model public transport networks as graphs. 

It includes notebooks for parsing, cleaning, and analyzing GTFS data, with support for multiple cities. 
Our code focuses on the Belgian and Dutch railways data. 

An example for the Chicago data is also provided.

The structure should look like this:
```bash
project/
├── INSTALL.md
├── README.md
├── config.py
├── imports.py
├── notebooks/
└── ...
---
```

## Cleaning data

The cleaning process involves several steps to ensure the GTFS data is suitable for graph modeling. 
The main steps are described below:
- notebook 1
- notebook 2
- ...

## Optimization

Five different versions of the functions are provided to optimize the cleaning process.

```python
get_all_GTC
```

```python
P_space
```

## 📁 Project Structure
```bash
├── notebooks/
│   ├── Belgium Railways.ipynb # Main notebook for working on the Belgian Data
│   ├── CheckNodes&Routes.ipynb # Notebook to check, visualize and analyze the L-Graph
│   ├── Chicago.ipynb # Main notebook for working on the Chicago Data
│   ├── DeleteNodes&Routes.ipynb # Notebook focussing on cleaning the L-Graph by deleting unwanted nodes and routes
│   ├── MergeRoutes.ipynb # Notebook which merges direct routes with the actual path. 
│   └── P-Space.ipynb # Notebook to work on the P-Graph being generated from the cleaned L-Graph
│   └── TopologicalIndicators_Belgium.ipynb # Notebook to work on the topological indicators of the Belgian Railways
│   └── PTopologicalIndicators_Netherlands.ipynb # Notebook to work on the topological indicators of the Dutch Railways
│
├── data/pkl/
│   ├── belgium_nodesCleaned.pkl # L-Graph after cleaning the nodes
│   ├── belgium_routesCleaned.pkl # # L-Graph after cleaning the routes after the nodes
│   ├── belgium_P.pkl # P-Graph
│   ├── belgium.pkl # Original L-Graph of Belgian Railways
│   └── chicago.pkl # Original L-Graph of Chicago Metro
│   └── gtc_data.pkl # Consists of the GTC output
│   └── nl_merged.pkl # Cleaned L-Graph of Dutch Railways
|
├── data/sqlite/
│   ├── belgium.sqlite 
│   └── chicago.sqlite
│
```

## SQLite Files

Download the SQLite database files required for the notebooks from the following link:

[Download sqlite.zip](https://www.dropbox.com/scl/fi/hd4l1vxb43j10tglrl4x5/sqlite.zip?rlkey=htpb057n5ibygd0p1iyldn42z&st=2xrzsyo5&dl=0)

### Setup Instructions

1. Download the `sqlite.zip` file from the link above.  
2. Extract the contents of the zip file.  
3. Move the extracted folder into the `data` directory of your project.  

After extraction, the structure should look like this:
```bash
project/
├── data/
│   ├── pkl/
│   └── sqlite/
│       ├── belgium.sqlite
│       └── chicago.sqlite

