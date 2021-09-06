## Pre-trained models

In this folder we will describe each pre-trained model,
how it was developed and it's appropriate usage

1. `xgb_basic.pkl`: This model utilizes the preprocessed SDF features, and fits to
PAXDB protein abundance measured in parts-per-million. PAXDB data is preprocessed by multiplying
   by 1 million and transforming using log-2 to get a normal distribution. SDFs with HGNC labels
   and PAXDB data are merged together, whereby we fit a hyperparameter search over learning rate using
   XGBoost. The best model is then re-fitted and output. See `Scripts/Develop_Pretrained_Models.ipynb` for
   code. 