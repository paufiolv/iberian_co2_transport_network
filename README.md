# Iberian CO<sub>2</sub> Transport Network

This repository contains the code used to conduct a study of a cost-optimal CO<sub>2</sub> transport network for CCUS in the Iberian Peninsula. To this end, a stochastic mixed-integer program has been developed that accounts for uncertainty in CO<sub>2</sub> demand for industrial utilization, identifying both the optimal network layout and its phased development over time. Three scenarios with different utilization forecasts are defined, and the impact of uncertainty on the net cost to the transport network operator is analyzed. The proposed approach explicitly incorporates pipeline pressure constraints, including friction and elevation losses, enabling more realistic and efficient infrastructure planning. The results help identify robust configurations under uncertainty and offer valuable insights for the development of the regional CCUS infrastructure. The study, developed in Python and optimized using Gurobi, uses source data from Excel sheets to generate new Excel sheets with results. Several additional Python programs are included to obtain some relevant figures.

## 1) Repository structure

- **data/** contains, one the one hand, the .xlsx files with the techno-economic parameters used for the model, and on the other hand, the .json files with the geography of the studied region.
- **results_processing/** contains the .py files required to generate different plots to analyze the results provided by the model execution.
- **iberian_co2_network/** contains the .py files required to execute the optimization models (the developed and the traditional one).
- **releases** contains the SRTM (Shuttle Radar Topography Mission) file, which includes an accurate elevation profile of the Iberian Peninsula, also required to obtain part of the data.

## 2) Requirements and environment

The project has been developed using:

- Python **v. 3.13**.
- **Libraries:** Pyomo, pandas, geopandas, numpy, matplotlib, CoolProp, shapely, collections, math, ast, future, typing, sys, os, pathlib, rasterio, pyproj, scipy.
- Solver: Gurobi **v. 11.0.3**.
- Computer operating system: **Windows 11**.

## 3) Installation

1. Create a virtual environment
2. Install all the required libraries
3. Run the model

## 4) How to run (reproducibility)

## 5) Data

## 6) Expected output
