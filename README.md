# Cold-Start-Retail-Forecasting
An end-to-end Data Science and AI pipeline solving the cold-start retail problem using LLM augmentation and neural networks.

# LLM-Augmented Retail Demand Forecasting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch/TensorFlow](https://img.shields.io/badge/framework-PyTorch%20%7C%20TensorFlow-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains a graduation thesis project designed to solve the **cold-start problem** in retail demand forecasting. Traditional forecasting models struggle when introducing new products with zero historical sales data. This project bridges that gap by combining **Generative AI (LLMs) for synthetic data augmentation** with **Deep Learning (LSTM/GRU) time-series forecasting architectures**.

By utilizing Large Language Models to generate synthetic historical contexts (accounting for 50% of the training dataset), the neural networks can learn underlying demand patterns for entirely new SKUs before they hit the shelves.

## Key Results & Business Value
* **27% Predictive Improvement:** Outperformed traditional baseline models by 27% in cold-start scenarios.
* **Evaluation Metrics:** Validated using walk-forward validation with **sMAPE** (Symmetric Mean Absolute Percentage Error) and **WAPE** (Weighted Absolute Percentage Error).
* **Data Augmentation:** Successfully integrated prompt engineering to synthesize realistic retail demand data, mitigating the sparsity of cold-start datasets.

## Methodology & Architecture
1. **Data Pipeline & Augmentation:** Historical sales data is processed, and LLMs are prompted to generate synthetic but highly correlated demand patterns for new items, expanding the dataset by 50%.
2. **Model Training (Deep Learning):**
   * **LSTM (Long Short-Term Memory):** Captures long-term seasonal trends and demand spikes.
   * **GRU (Gated Recurrent Unit):** Offers a computationally efficient alternative with highly competitive accuracy.
3. **Walk-Forward Validation:** Simulates real-world deployment by training on past data and predicting strictly future time steps, ensuring robust metric evaluation without data leakage.

## Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch / TensorFlow (Keras)
* **Data Science & ML:** Scikit-learn, Pandas, NumPy
* **Generative AI:** LLM Prompt Engineering / Synthetic Data Generation
* **Visualization:** Matplotlib, Seaborn

## Project Structure
```text
├── data/
│   ├── raw/               # Original cold-start retail datasets (Not uploaded to Git)
│   └── synthetic/         # LLM-generated synthetic data pipelines
├── models/                # Saved LSTM & GRU model weights (.h5 / .pt)
├── notebooks/             # Exploratory Data Analysis (EDA) & baseline testing
├── src/
│   ├── augment.py         # LLM prompt engineering & data synthesis scripts
│   ├── train.py           # Deep learning model training loops
│   ├── evaluate.py        # sMAPE and WAPE metric calculations
│   └── utils.py           # Helper functions for walk-forward validation
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
How to Run
1. Clone the repository

Bash
git clone [https://github.com/your-username/Cold-Start-Retail-Forecasting.git](https://github.com/your-username/Cold-Start-Retail-Forecasting.git)
cd cold-start-retail-forecasting
2. Create a virtual environment and install dependencies

Bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
3. Run the training pipeline

Bash
python src/train.py --model lstm --epochs 50
4. Evaluate the model

Bash
python src/evaluate.py --weights models/lstm_best.pt
Contact
Kadir Onur Hancı

Computer Science & Data & AI

[[LinkedIn Profile Link](https://www.linkedin.com/in/kadir-onur-hancı-01a197231/?isSelfProfile=true)] | [Email Address] (hancikadironur@gmail.com)
