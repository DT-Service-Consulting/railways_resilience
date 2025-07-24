# Public Transport Network Graph Modelling using GTFS Data

This repository provides tools for processing GTFS (General Transit Feed Specification) data to model public transport networks as graphs. It includes notebooks for parsing, cleaning, and analyzing GTFS data, with support for multiple cities. Our code focuses on the Belgian data, particularly rail lines. There is also some work on the Chicago data (provided by Dr. Renzo Massobrio (renzo.massobrio@uantwerpen.be)), which we took as a reference to while working on the Belgian data.
The structure should look like this:
```
project/
├── INSTALL.md
├── README.md
├── config.py
├── imports.py
├── notebooks/
└── ...

---
```


### Run the Jupyter Notebook
Activate the environment and start Jupyter:
```bash
jupyter notebook
```
Select a notebook from the list (e.g., Belgium Railways.ipynb) to begin your analysis.

## 📁 Project Structure
```python
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
