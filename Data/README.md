## Data source

In this folder we will describe each of the data files. Before you can access them, you will need
to unzip the package file using 7-zip.

1. `feature_desc.xlsx`: This excel spreadsheet contains a description of each of the sequence-derived
features.
2. `hgnc_labels.csv`: A spreadsheet containing label mappings between protein, mRNA and gene databases
3. `PAXDB_WHOLE_CONSENSUS.csv`: PAXDB database average steady-state protein abundances for H. sapiens across all tissues
4. `sdf_unscaled.csv`: Unscaled sequence-derived features (SDFs) dataset.
5. `sPCA_dg_mca.csv`: Scaled and preprocessed sequence-derived features (SDFs) by PCA and MCA   
6. `xgb_basic_predictions.csv`: A spreadsheet containing predicted values using `xgb_basic.pkl` model and 
`PAXDB_WHOLE_CONSENSUS.csv` as target.