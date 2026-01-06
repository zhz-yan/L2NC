# Demo

This repository provides the demo code for the experiments presented in our paper.  
All scripts are located in the `src/` directory.

## Directory Structure

- `src/exp_5_1.py`: Reproduces Scenario 1 (Single Unknown Appliance) results (corresponding to Section 5.1 in the paper)
- `src/exp_5_2.py`: Reproduces Scenario 2 (Multiple Unknown Appliances / Openness) results (corresponding to Section 5.2 in the paper).
- `src/results/`: All experimental results will be saved here in `.json` format
- `src/data/`, `src/features/`: Contains the pre-processed feature data required for the experiments.

## How to Run

You can replicate the experimental results reported in the paper using the following commands:

#### 1. Reproduce Single Unknown Scenario (Scenario 1)
This script runs the evaluation where one appliance type is treated as unknown at a time.

  ```bash
  python src/exp_5_1.py
  ```

#### 2. Reproduce Varying Openness Scenario (Scenario 2)
This script evaluates the model under different degrees of openness (varying number of unknown classes).

  ```bash
  python src/exp_5_2.py

## Output
After running the scripts, the evaluation metrics will be printed to the console and saved to the src/results/ directory for analysis.


## Requirements

To reproduce the results, please ensure you have installed the following dependencies:
- `numpy`
- `scikit-learn`
- `matplotlib` (if used for plotting)

You can install them via pip:
```bash
pip install numpy scikit-learn
