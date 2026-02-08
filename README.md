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

## 4) Data


## 5) How to run

### Developed model

### Traditional model

## 6) Expected output

The optimization model must be run for each value of theta. The resulting files are an Excel file and 15 figures representing the development of the CO2 transport network:

- The resulting Excel file, named "stochastic_results_theta_x.xx" or "stochastic_traditional_results_theta_x.xx" (depending of whether the developed or the traditional model has been run) is stored in a folder called “results_summaries” and contains 13 sheets of results. For each scenario (LUS, EUS and HUS), there are four sheets:
  - The first sheet contains the finally selected candidate pipes, their diameters, the pressure gradient through them, and the flow transported in each time step.
  - The second one includes the remaining capacity in all the sinks after each time step.
  - The thrid sheet shows the amount of CO2 utilized along each time step in each one of the utilization nodes.
  - The forth one includes the breakdown of all costs considered by the model, as well as the profits from the sale of CO2 for its industrial utilization.
  - The last sheet is common for all the scenarios and includes a comparison between the length of the transport network deployed in each scenario for each time step of the model.
 - The 15 figures obtained are named "developed_solution_XXX_theta_Y.YY_ZZZZ.png", where XXX is the scenario, Y.YY is the value of theta and ZZZZ is the time step of the plot. These 15 figures are saved in a folder called "output_developed_plots_theta_Y.YY", which in turn is stored in the folder "output_developed_plots". In the case of the traditional model, the 15 figures obtained are named "stochastic_traditional_solution_XXX_theta_Y.YY_ZZZZ.png". These 15 figures are saved in a folder called "output_stochastic_traditional_theta_Y.YY", which in turn is stored in the folder "output_traditional_plots".

## 7) Results post-processing and figures generation

The files contained in the folder "results_processing" allow one to generate insightful figures from the results, which need to have been previously obtained for the desired value of theta.

- **plots_from_results.py**: The file generates 15 figures (3 scenarios x 5 time steps) representing the temporal deployment of the CO2 transport network according to the scenario. The results obtained in the time steps 2030, 2035, 2040, 2045, and 2050 are shown for the LUS, EUS, and HUS scenarios. To change the value of theta, find the line “THETA = 1.00 # <-- Change this value manually each time you run the script” and modify the value.
- **scenario_plots_together.py**: Generates the same results as “plots_from_results.py,” but includes the entire time development for that scenario in a single figure (includes the plots of the 5 time steps in a single figure). Therefore, it generates 3 figures, one for each scenario. As in the previous case, to change the value of theta, find the line “THETA = 1.00 # <-- Change this value manually each time you run the script” and modify the value.
- **comparison_with_traditional_model_plots.py**: It generates, for the desired theta value and scenario (to be defined manually in the code itself), a figure comparing the transport network deployment obtained between the developed model and the traditional model for the time steps 2030, 2040, and 2050.
- **system_cost_per_ton_plot.py**: Generates, for a given scenario (to be selected manually in the code itself), a figure with the Levelized Cost of CO2 Transport based on the value of theta. The total cost is displayed based on the contributions of each process (capture, pipeline transport, boosting, injection, etc.).
- **system_operator_costs_plot.py**: For a given scenario (to be selected manually in the code itself), it generates a figure containing, for each value of theta, the total deployed cost of the CO2 network. It also shows the profits generated by the sale of CO2 for industrial use, as well as the net cost of the system, resulting from subtracting the profits from the total cost.
- **theta_comparison_plots.py**: Given a scenario, a time step, and two values of theta (to be defined manually in the code), it generates a figure with two plots of the transport network deployment obtained for each selected value of theta.
- **traditional_vs_developed_costs_plot.py**: It generates, for the desired theta value and scenario (to be defined manually in the code itself), a figure comparing the transport network costs between the developed model and the traditional model.
