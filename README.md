# Bioassay Analysis Using the Metropolis Algorithm

## Overview
This project involves the application of the Metropolis algorithm, a foundational MCMC technique, to analyze data from a bioassay experiment. The goal is to estimate the posterior distributions of two parameters in a non-conjugate model, which were initially explored using a grid approximation method in the referenced literature.

## Repository Structure
- `metropolis_algorithm.py`: Contains the Python code implementing the Metropolis algorithm along with the necessary statistical functions for the bioassay analysis.
- `data_analysis.ipynb`: Jupyter notebook detailing the analysis process, including model specification, likelihood function computation, and convergence diagnostics.
- `data/`: Directory containing the dataset used in the analysis.
- `figures/`: Directory containing visualizations such as trace plots generated during the analysis.
- `report/`: Contains the LaTeX source code for the final analysis report along with the compiled PDF document.

## Key Findings
- The acceptance rates for the MCMC chains post burn-in suggest a healthy degree of parameter space exploration.
- R-hat diagnostics indicate that the parameter `alpha` has likely converged to the target distribution, while `beta` may require further analysis.
- Trace plots visually corroborate the convergence diagnostics and highlight the areas for potential improvement in the sampling process.



## Requirements
This analysis was performed using Python 3.x. Required packages include `numpy`, `scipy`, `matplotlib`, and `arviz`. They can be installed via pip:
```bash
pip install -r requirements.txt
