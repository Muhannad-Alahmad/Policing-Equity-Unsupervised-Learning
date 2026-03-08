# Policing Equity: Unsupervised Learning on Field Interview Data (2011–2015)

This repository contains the code and exported figures for **Task 2: Policing Equity** (Machine Learning – Unsupervised Learning and Feature Engineering).

The analysis uses **unsupervised learning** (PCA + K-Means) to explore patterns in policing field interviews and to support a more informed discussion about policing activity patterns.

## Links
- **Kaggle Notebook (runnable):** https://www.kaggle.com/code/muhannadalahmad/policing-equity-unsupervised-learning
- **Dataset (Kaggle):** Data Science for Good: *Center for Policing Equity*

  https://www.kaggle.com/center-for-policing-equity/data-science-for-good

## What’s in this repo
- A Jupyter/Kaggle notebook with the full pipeline (loading → cleaning → EDA → PCA → clustering → interpretation)
- A Python script export of the notebook
- Exported plots (EDA, PCA, elbow/silhouette, cluster visualizations)

## Notebook sections (quick guide)
The notebook is organized into these sections:

1. Setup and Data Loading  
2. Initial Exploration  
3. Data Cleaning  
4. Exploratory Data Analysis (EDA)  
5. Feature Engineering  
6. Dimensionality Reduction (PCA)  
7. Clustering (K-Means)  
8. Results and Cluster Interpretation  
9. Summary  

## Dataset note (not included here)
The dataset is **not included** in this GitHub repository.

To run locally:
1. Download the dataset from Kaggle (link above).
2. Place the file `11-00091_Field-Interviews_2011-2015.csv` somewhere on your machine.
3. Update the `FILE = ...` path in the notebook/script to point to your local file.

On Kaggle, the notebook reads directly from `/kaggle/input/...`.

## Quick results (from the Kaggle run)
- Records after cleaning/filtering to 2011–2015: **150,775**
- Encoded feature matrix: **28 features** (one-hot encoding of selected variables)
- Dimensionality reduction: PCA (used for visualization and clustering input)
- Clustering: K-Means (k chosen via elbow + silhouette)

## Project structure
```text
policing-equity-case-study/
├── README.md                              # Project overview and instructions
├── requirements.txt                       # Python dependencies
│
├── notebook/
│   └── policing-equity-unsupervised-learning.ipynb   # Main Kaggle/Jupyter notebook
│
├── code/
│   └── policing-equity-unsupervised-learning.py      # Python script version of the notebook
│
└── outputs/
    └── figures/                           # Generated plots (exported from Kaggle)
        ├── eda_visualizations.png         # EDA: race, district, year, age
        ├── pca_results.png                # PCA: cumulative variance + 2D projection
        ├── clustering_optimization.png    # Elbow method + silhouette scores
        └── cluster_visualizations.png     # Clusters in PCA space + race distribution by cluster
```

## Installation (local)
```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run (local)
### Option A: Notebook
```bash
jupyter notebook notebooks/policing-equity-unsupervised-learning.ipynb
```

### Option B: Python script
```bash
python code/policing-equity-unsupervised-learning.py
```

> Tip: The script is an export of the notebook. If you update the notebook, consider re-exporting to keep them in sync.

## Reproducibility
- The easiest way to reproduce results is via the **Kaggle notebook**, since the dataset is already mounted as input there.
- If you run locally, ensure your dataset path is correct and rerun from the top.

## Acknowledgments
- Center for Policing Equity for providing the dataset (via Kaggle)
- IU course: *Machine Learning – Unsupervised Learning and Feature Engineering*
