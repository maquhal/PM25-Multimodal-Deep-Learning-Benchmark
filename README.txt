# Deep Learning & Graph Neural Networks Project

This project explores several deep learning architectures, including CNNs, RNNs, LSTMs, and Graph Neural Networks (GNNs), for predictive modeling, evaluation, and feature learning.  
The implementation is provided in a Jupyter Notebook, which can be run step by step once all dependencies are installed.

---

## Project Overview

The notebook demonstrates:
- **Data preprocessing** (scaling, normalization, statistical correlation).  
- **Classical Deep Learning models** with TensorFlow/Keras:
  - Convolutional Neural Networks (CNNs).  
  - Recurrent Neural Networks (RNNs).  
  - Long Short-Term Memory networks (LSTMs).  
  - Bidirectional LSTMs.  
- **Graph Neural Networks (GNNs)** with PyTorch Geometric:
  - Graph Convolutional Networks (GCN).  
  - Graph-level pooling and feature learning.  
- **Evaluation metrics** for regression and classification tasks:
  - Mean Squared Error (MSE).  
  - Mean Absolute Error (MAE).  
  - R² score.  
  - Pearson correlation coefficient.  

---

# Requirements

You’ll need Python **3.8+** and the following libraries:

- **Core libraries**:  
  - `pandas`, `numpy`  
- **Visualization**:  
  - `matplotlib`, `seaborn`  
- **Preprocessing & metrics**:  
  - `scikit-learn` (`MinMaxScaler`, metrics)  
  - `scipy` (`pearsonr`)  
- **Deep Learning (TensorFlow/Keras)**:  
  - `tensorflow` (with `keras`)  
- **Deep Learning (PyTorch & PyTorch Geometric)**:  
  - `torch`, `torch_geometric`

### Install all dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy tensorflow torch

#### For Mamba 

pandas
numpy
scikit-learn
scipy
matplotlib
tqdm
torch
mamba-ssm
transformers
onnx
