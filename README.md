# MultimodalEELS-XAS

This repository contains code and data for analysis of multimodal spectroscopies (EELS and XAS) on NMC cathode materials. It includes simulated data, experimental datasets, and Python modules for featurization, dimensionality reduction, and machine-learning model.

---

## Data

- **Simulated Data**
  - `Data/NMC_fdmnes.json`  
    A list of simulated spectra. Each entry is a dictionary with:
    - `composition`: e.g. NMC811
    - `structure_id`: structure identifier
    - `structure`: pymatgen `Structure` object
    - `O`, `Ni`, `Mn`, `Co`: corresponding simulated spectra (energy, intensity)
  - `Data/O_in.txt`, `Data/Ni_in.txt`, `Data/Mn_in.txt`, `Data/Co_in.txt`
    - Fdmnes input files for simulation

- **Experimental Data**
  - `Data/XAS/`  
    - Contain O K-edge and transition metal L-edge XAS spectra
    - Excel sheet `NCM MW & TC.xlsx` used to calculate Li content from capacity
  - `Data/EELS/`  
    - Contains EELS spectra of 5% Si-doped NMC.
    - Subfolder `/TM/`  
      - Transition metal (Ni, Mn, Co) EELS data.
    - Subfolder `/Si/`  
      - Si EELS data.

---

## Dependencies

- Python 3.7+  
- Core packages:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `scikit-learn`
  - `xgboost`
  - `hyperopt`
  - `pymatgen`

---

## Code Overview

- **utils.py**  
  General helper functions for data handling:  
  - Load spectral data from multiple file types (`.dat`, `.txt`, `.csv`, `.msa`, etc.).  
  - Read Fdmnes output.  
  - Interpolate spectra.
  - Generate artificial noise (Gaussian and Lorentzian).
  - Apply Lorentzian broadening.  

- **featurization.py**  
  Functions to build descriptors from spectra:  
  - **CDF** (cumulative distribution function) of intensity.  
  - **Piecewise cubic polynomial fits** for local shape descriptors.  
  - **Gaussian mixture decomposition** for multi-peak fitting.  

- **dim_reduction.py**  
  Dimensionality reduction methods and plotting helpers:  
  - **PCA** with explained variance plotting and inverse transforms.  
  - **t-SNE** for non-linear embeddings.  
  - 2D/3D embedding visualization utilities.  

- **train.py**  
  Training pipeline with Bayesian hyperparameter optimization:  
  - `BOhyper_reg` class wraps XGBoost regression with **Hyperopt**.  
  - Performs **k-fold CV** for robust evaluation.  
  - Tracks best model (lowest RMSE).  
  - Includes functions to evaluate on held-out test data.  

---

## Testing & Workflow Guidance

1. **Load training data**  
   - Use functions in `utils.py` to load simulated (`NMC_fdmnes.json`) or experimental spectra (XAS/EELS).  

2. **Featurize spectra**  
   - Apply CDF, polynomial descriptors, or Gaussian mixture decomposition with `featurization.py`.  

3. **Apply dimensionality reduction**  
   - Run PCA, t-SNE, or UMAP using `dim_reduction.py` to compress features into lower dimensions.  

4. **Train regression model**  
   - Use `train.py` to run Bayesian optimization (`BOhyper_reg`) over XGBoost hyperparameters.  
   - Save the best trained model.  

5. **Test new data**  
   - Load test spectra the same way as training.  
   - Apply **the same featurization steps** and **the same dimension reduction model** (e.g. PCA fitted on training data).  
   - Run the trained model:  
     ```python
     preds = best_model.predict(X_test_transformed)
     ```  

6. **Evaluate performance**  
   - Compute RMSE and R² with `evaluate()` in `train.py`.  
   - Compare predictions with ground truth (capacity, Li content, etc.).  

