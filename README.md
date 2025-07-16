# LightWeight-Acoustic-CNN
This repository provides a recognition model for disaster-related acoustic signals.  
It demonstrates how to preprocess acoustic signals, generate time-frequency representations, 
and train a modified deep learning model for binary classification tasks.  

A small portion of audio recordings and pre-trained model weights are provided 
for demonstration purposes.
## File structure
```

## File structure

LightWeight-Acoustic-CNN
├── data/ # Example audio files and spectrograms
├── features/ # Preprocessed .npy and .png files
├── preprocessing.py # Convert .wav to spectrograms
├── model.py # Modified  LightWeight CNN model
├── train.py # Train the binary classification model
├── test.py # Run inference with trained model
├── requirements.txt
└── README.md

```

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

