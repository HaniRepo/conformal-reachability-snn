# Conformal Reachability Analysis with SNN-Based Monitoring

This repository provides a reproducible artifact for uncertainty-aware predictive maintenance using:

- Spiking Neural Networks (SNN)
- Conformal Prediction
- Discrete-Time Markov Chains (DTMC)
- Formal verification via Storm model checker

The framework transforms raw sensor signals into **probabilistic safety guarantees**, enabling reasoning about system degradation and failure risk.

---

##  Quick Start

Run the interactive artifact:

```bash
python run_artifact.py
```
You will see a menu:
```text
1→ CWRU case study  
2 → Filtration case study  
3 → NASA case study  
4 → Fusion analysis  
5 → Extended filtration  
0 → Exit
```
---
## Method Overview

The pipeline is:

- Signal preprocessing and feature extraction
- SNN-based degradation scoring
- Conformal prediction → uncertainty intervals 
- State discretization 
- DTMC construction 
- Buildling automatic Formal models
- Calculating finite-horizon reachability

---
## Environment Setup
You can use the environment where we freezed simply by running:
```bash
conda env create -f environment.yml
conda activate qest26
```

Alternatively, you can install related packages in the requirement file.

---
## Running Experiments
🔹 Option 1 (Recommended)
```bash
python run_artifact.py
```
🔹 Option 2 (Manual)
```bash
python analyze_cwru.py
python analyze_filter.py
python analyze_nasa.py
python analyze_fusion.py
python analyze_extended_filter.py
```
---
##  Data, Outputs, and Reproducibility

###  Datasets

This repository supports multiple industrial case studies:

- **CWRU Bearing Dataset**  
  https://engineering.case.edu/bearingdatacenter  

- **NASA C-MAPSS Dataset**  
  https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data  

- **Filtration Dataset**  
  Included within the repository (`data/data_filter` )



---

### Outputs

All experiment outputs are organized into dedicated folders:

```text
results_cwru/
results_filtration/
results_extended_filtration/
results_nasa/
```
---
Each folder contains:

- Prediction intervals (conformal outputs)
- DTMC transition matrices
- Finite-horizon reachability probabilities
- Visualization plots
- Generated Storm model files

---

###  Formal Verification

Probabilistic verification is performed using the **Storm model checker**:

https://www.stormchecker.org/

Generated artifacts include:

- `.pm` files (DTMC models)
- Storm command scripts
- Reachability and analysis summaries

---

###  Key Contributions

- Distribution-free uncertainty quantification via conformal prediction  
- Data-driven abstraction into discrete-time Markov chains (DTMCs)  
- Formal probabilistic guarantees through model checking  
- Validation across multiple datasets (CWRU, NASA, Filtration)  

---

### ⚠️ Notes

- Storm must be installed separately or you can use our docker that will be released shortly.  

---

###  Reproducibility

This artifact has been tested on:

- Python 3.10  
- Conda-based environment (`environment.yml`)  
- Windows and Linux platforms  
---


