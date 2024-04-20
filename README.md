# Barcode-GCN: A Graph Convolution Framework for DNA Taxonomic Classification

This repository contains code for training and evaluating the proposed method in the paper.

### 1- Clone this repo:
``` bash
git clone https://github.com/Niousha12/BarcodeGNN.git
cd BarcodeGNN
```

### 2- Install requirements:

Create and activate a virtual environment:
```bash
python -m venv /path/to/new/virtual/environment
source venv/bin/activate 
```

Install requirements:
```bash
pip install -r requirements.txt
```
Also, from https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html, select the configuration of your system and install `torch_geometric`.



### 3- Train the model:
Start the train by the following command.

``` bash
python main.py
```


### 4- Evaluate the trained model:
Start the evaluation by the following command.

``` bash
python evaluation.py --checkpoint_path /path/to/your/checkpoint.pth
```
