# PM2.5 Multimodal Deep Learning Benchmark

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)
![MSc](https://img.shields.io/badge/MSc-Research-orange)

## Introduction

A research-oriented benchmark for 24-hour PM2.5 forecasting using multimodal environmental data and modern deep learning architectures.

This project was developed as part of my MSc dissertation in Artificial Intelligence at the University of Surrey. It integrates ground-based air-quality observations, meteorological reanalysis data, and satellite-derived aerosol information to evaluate multiple deep learning architectures in data-scarce regions.

## Research Objective

The project investigates how different deep learning architectures perform when forecasting PM2.5 concentrations using multimodal environmental data.

The study focuses on:

- Comparing baseline and advanced deep learning models.
- Evaluating performance across African and Middle Eastern cities.
- Assessing the contribution of satellite Aerosol Optical Depth (AOD) data.
- Identifying an effective model for 24-hour PM2.5 forecasting.

## Data Sources

The multimodal framework integrates:

- **Air-quality data:** PM2.5 observations from OpenAQ.
- **Meteorological data:** ERA5 reanalysis variables.
- **Satellite data:** Aerosol Optical Depth (AOD) from the Copernicus Atmosphere Data Store.

The selected cities include:

- Riyadh
- Dhahran
- Dubai
- Pretoria
- Potchefstroom
- Siavonga

## Models Evaluated

Seven deep learning architectures were evaluated:

1. Multi-Layer Perceptron (MLP)
2. One-Dimensional Convolutional Neural Network (1D-CNN)
3. Bidirectional Long Short-Term Memory (BiLSTM)
4. LSTM–RNN
5. Graph Neural Network (GNN)
6. Mamba State-Space Model
7. GNN–Mamba

## Experimental Setup

- Input sequence: 168 hourly observations
- Forecast horizon: 24 hours
- Temporal split:
  - 80% training
  - 10% validation
  - 10% testing
- Experiments conducted both with and without satellite AOD data.

## Evaluation Metrics

Model performance was evaluated using:

- Root Mean Squared Error (RMSE)
- Normalized Root Mean Squared Error (NRMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R²)
- Pearson Correlation Coefficient

## Key Results

- Evaluated seven deep learning architectures for 24-hour PM2.5 forecasting.
- 1D-CNN achieved the best overall forecasting performance.
- Multimodal learning outperformed single-source approaches.
- Satellite AOD data improved forecasting accuracy across multiple cities.
- Performance was assessed using RMSE, NRMSE, MAE, R², and Pearson Correlation.

## Repository Contents

## Requirements

- Python 3.8+
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- tqdm
- TensorFlow
- PyTorch
- PyTorch Geometric
- Transformers
- Mamba-SSM
- ONNX

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/maquhal/PM25-Multimodal-Deep-Learning-Benchmark.git
```

2. Enter the repository:

```bash
cd PM25-Multimodal-Deep-Learning-Benchmark
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Open the main notebook:

```bash
jupyter notebook Multimodal_PM25_Benchmark.ipynb
```

## Dissertation

**Title**

*A Benchmark of Deep Neural Network Models for PM2.5 Forecasting Using Multimodal Data: A Case Study in Africa and the Middle East*

University of Surrey, 2025

## Citation

If you use this repository in your research, please cite:

```bibtex
@mastersthesis{quhal2025pm25,
  author = {Mohammed A. Quhal},
  title = {A Benchmark of Deep Neural Network Models for PM2.5 Forecasting Using Multimodal Data: A Case Study in Africa and the Middle East},
  school = {University of Surrey},
  year = {2025},
  type = {MSc Dissertation}
}
```

## Author

**Mohammed A. Quhal**

- GitHub: https://github.com/maquhal
- LinkedIn: https://linkedin.com/in/maquhal
