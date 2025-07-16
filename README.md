# LightWeight-Acoustic-CNN
This repository provides a recognition model for disaster-related acoustic signals.  
It demonstrates how to preprocess acoustic signals, generate time-frequency representations, 
and train a modified deep learning model for binary classification tasks.  

A small portion of sub-audio (infrasound) recordings and pre-trained model weights are provided 
for demonstration purposes.
## File structure
data
└── 24h_10mmMAX_OI
    ├── allevents_dates.csv
    ├── models
    ├── study_area.csv
    ├── obs
    ├── OI_20152022_10mmMAX.csv
    ├── OI_raw_mask_piem_vda.csv
    ├── OI_regrid_mask_piem_vda.csv
    ├── OI_regrid_mask_piem_vda_unet.csv
    ├── OI_regrid_quota_unet.csv
    └── split

## Quick start

### 1. Install dependencies
    ```bash
    pip install -r requirements.txt

### 2. Preprocess example audio
    python preprocessing.py
### 3. Train the model
    python train.py

## Notes
This code is for demonstration and reproducibility only

Example data are generic infrasound samples for disaster monitoring, not the full research dataset

Some layer configurations and parameter tuning steps have been simplified
