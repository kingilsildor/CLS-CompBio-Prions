# CLS-CompBio-Seizures

This project simulates the diffusion of proteins and prions in a 2D grid, along with neuron health dynamics. The simulation generates various outputs, including plots and GIFs, to visualize the results.

## Prerequisites

Ensure you have [Conda](https://docs.conda.io/en/latest/) installed on your system. Conda will be used to create and manage the environment for this project.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Prions-Analysis
   ```

2. Create a Conda environment:
    ```bash
    conda create --name prions-analysis python=3.13
    ```
3. Activate the environment:
    ```bash
    conda activate prions-analysis
    ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Simulation
To run the simulation, execute the main.py file:
    ```bash
    python main.py
    ```
The simulation will generate results in the `results/` directory, including:
- Plots of protein, prion, and neuron concentrations over time.
- GIF animations of the simulation.
- Summary statistics and visualizations of neuron deaths.

## Cleaning Up
To remove temporary `.npy` files generated during the simulation, you can use the delete_npy function in `scripts/data_manipulation.py` or manually delete the `data/` directory.
Currently this function is being called in the main as well.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

