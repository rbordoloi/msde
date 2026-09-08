# Anomaly Detection via Mean Shift Density Enhancement (MSDE)

Official implementation of **Mean Shift Density Enhancement (MSDE)**, a fully unsupervised anomaly detection framework based on density-driven manifold evolution.

[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs10618--026--01244--5-blue)](https://link.springer.com/article/10.1007/s10618-026-01244-5)

## Overview

**MSDE** detects anomalies by observing how data points move under an iterative, density-driven mean-shift process. 
* Normal observations (in dense regions) remain relatively stable.
* Anomalous observations (in low-density regions) experience larger displacements.

By accumulating these displacements over multiple iterations, MSDE calculates a robust **anomaly score**. 

> **Note on Naming:** You will see the name `MSML` in some source files (e.g., `MSML_v10.py`). This is a legacy project name from development. The method is officially known as **MSDE**.

## Repository Highlights

* `MSDE_implementation.py` - **Standalone MSDE Implementation** (Recommended for general use and integration).
* `MSML_v10.py` - **Official Paper Implementation** (The exact code used in the research publication).
* Jupyter Notebooks - Extensive experiments covering hyperparameter optimization (Optuna), scalability, and credit card fraud detection.

## Installation

A Python environment using **Python 3.12** is recommended.

1. **Clone the repository and set up a virtual environment:**
   ```bash
   conda create -n msde python=3.12
   conda activate msde
## Usage

**To run the standalone MSDE implementation:**
```bash
python MSDE_implementation.py
```
## Citation

If you use MSDE or this code in your research, please cite our paper published in *Data Mining and Knowledge Discovery* (2026):

```bibtex
@article{kar2026anomaly,
  title={Anomaly detection via mean shift density enhancement},
  author={Kar, Pritam and Bordoloi, Rahul and Wolkenhauer, Olaf and Bej, Saptarshi},
  journal={Data Mining and Knowledge Discovery},
  volume={40},
  number={5},
  articleno={77},
  year={2026},
  publisher={Springer},
  doi={10.1007/s10618-026-01244-5}
}
