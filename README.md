# 5G Interference Analysis using Machine Learning

This final year project analyzes RF IQ signal data and predicts 5G interference severity using machine learning. It extracts statistical features from I/Q samples, creates synthetic 5G interference parameters, trains a Random Forest classifier, and saves evaluation plots.

## Features

- Loads RF signal data from `RML2016.10a_dict.pkl`
- Extracts RMS, variance, skewness, and kurtosis features
- Simulates 5G interference parameters such as frequency, power, bandwidth, SIR, and gain compression
- Classifies interference severity into three classes
- Reports accuracy, macro F1 score, classification report, and ROC-AUC
- Saves signal, severity distribution, confusion matrix, and feature importance plots

## Project Structure

```text
.
|-- main.py
|-- Mbu_finalyear_project.py
|-- Mbu_finalyear_project.ipynb
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Requirements

- Python 3.10+
- RML2016.10a dataset file: `RML2016.10a_dict.pkl`

The dataset file is not committed to GitHub because it is large. Place it in the project root before running the script, or pass a custom path with `--dataset`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Optional arguments:

```bash
python main.py --dataset path\to\RML2016.10a_dict.pkl --output-dir outputs --sample-limit 30000
```

To open the notebook:

```bash
jupyter notebook Mbu_finalyear_project.ipynb
```

## Outputs

Generated plots are saved in the `outputs/` directory:

- `signal_plot.png`
- `severity_distribution.png`
- `confusion_matrix.png`
- `feature_importance.png`

## Author

Yashwanth Varakuti
