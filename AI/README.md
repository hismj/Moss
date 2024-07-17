# MCADS: Lightweight Android Malware Detection System

## Overview
MCADS is a lightweight and efficient system for detecting malicious Android applications. It uses a two-layer structure:
1. The first layer analyzes AndroidManifest.xml files to create feature vectors for classification using an enhanced MLP network.
2. The second layer converts classes.dex files into RGB images for classification using an enhanced ShuffleNetV2 network.

## Project Structure
MCADS/
├── data/
│ ├── extract_manifest.py
│ ├── create_xml_db.py
├── models/
│ ├── init.py
│ ├── mlp.py
│ ├── shufflenetv2_eca.py
├── utils/
│ ├── init.py
│ ├── xml_vectorizer.py
│ ├── dex_to_image.py
│ ├── metrics.py
├── train.py
├── test.py
├── config.py
├── requirements.txt
└── README.md
## Setup
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   Extract AndroidManifest.xml files from APKs
   python data/extract_manifest.py
   Create XML database:
   python data/create_xml_db.py
   
## Training
1. To train the MLP model on the XML vectors:
    ```bash
    python train.py
    
## Testing
1. To test the MLP model on the XML vectors: 
   ```bash
   python test.py
## Configuration
1. 
    ```bash
    Modify config.py to change paths and parameters as needed.

# LICENCE
MIT
### Requirements File

#### requirements.txt
  torch
  torchvision
  Pillow
  scikit-learn
  umpy

With this structure and code, you should be able to build, train, and test your MCADS system in a modular and organized manner. The provided scripts handle data extraction, XML database creation, MLP model definition and training, and utility functions for vectorizing XML and converting DEX files to images. Make sure to fill in any missing pieces and adjust paths as necessary for your specific use case.


# TODO 
#### 最好模型选择

