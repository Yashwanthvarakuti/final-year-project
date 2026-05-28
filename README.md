# Machine Learning–Driven Prediction and Detection of 5G-Induced Nonlinear Interference in Radio Altimeters

## Overview

This final year project presents a machine learning–based framework for predicting and detecting 5G-induced nonlinear interference in aviation radio altimeters using RF IQ signal analysis and spectral feature engineering.

The rapid deployment of mid-band 5G communication systems has introduced significant RF coexistence challenges for aeronautical radio altimeters operating in the 4200–4400 MHz band. High-power adjacent-band 5G emissions can produce nonlinear distortion, LNA saturation, IF gain compression, and altitude estimation errors in radio altimeters.

This project combines RF signal processing, feature engineering, and machine learning to classify interference severity levels and predict height-error behavior in real time.

The work was also presented as part of an IEEE conference-level academic project presentation.

---

# Key Features

* RF IQ signal analysis using DeepSig RML2016.10a dataset
* Time-domain and frequency-domain feature extraction
* Simulation of 5G interference parameters
* Random Forest and SVM-based interference classification
* Height-error prediction using regression models
* Detection of severe interference and NCD-risk conditions
* Automated visualization dashboard generation
* Real-time monitoring logic for aviation safety analysis

---

# Technical Highlights

## RF Feature Engineering

Extracted statistical and RF-domain features including:

* RMS
* Variance
* Skewness
* Kurtosis
* Peak amplitude
* IF peak power
* Spectral centroid
* Noise-floor elevation
* 3 dB bandwidth
* Signal-to-Interference Ratio (SIR)
* Gain compression ratio

---

# Machine Learning Pipeline

The project performs:

1. RF dataset preprocessing
2. Feature extraction from IQ samples
3. Synthetic 5G interference simulation
4. Severity classification
5. Height-error prediction
6. Visualization and dashboard generation

### Classification Targets

* No Interference
* Moderate Distortion
* Severe Saturation

---

# Model Performance

| Metric                   | Value |
| ------------------------ | ----- |
| Classification Accuracy  | 98.4% |
| Severe-Class Sensitivity | 95.1% |
| Moderate-Class F1 Score  | 0.97  |
| Height Error RMSE        | 0.37% |
| Height Error R²          | 0.96  |

---

# Project Architecture

Dataset Construction & Preprocessing
→ Feature Extraction
→ ML Model Development
→ Interference Detection & Height-Error Prediction
→ Decision-Level Safety Monitoring

---

# Project Structure

```bash
.
│── main.py
│── Mbu_finalyear_project.py
│── Mbu_finalyear_project.ipynb
│── run_project.bat
│── requirements.txt
│── .gitignore
│── README.md
│
└── outputs/
    │── signal_plot.png
    │── severity_distribution.png
    │── confusion_matrix.png
    │── feature_importance.png
    └── dashboard.html
```

---

# Dataset

This project uses the DeepSig RadioML dataset for RF signal analysis.

Dataset sources:

* Official DeepSig Reference
  https://github.com/ianblenke/deepsig_dataset

* Download Mirror
  https://zenodo.org/records/18397070

Supported dataset files:

* RML2016.10a_dict.pkl
* RML2016.10a_dict_optimized.pkl

Note:
Dataset files are not included in this repository because of large file size limitations.

Place the dataset file in the project root directory before execution.

---

# Requirements

* Python 3.10+
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* SciPy
* Seaborn
* Jupyter Notebook

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Installation

## Clone Repository

```bash
git clone <your-repository-link>
cd <repository-name>
```

## Create Virtual Environment

```bash
python -m venv .venv
```

## Activate Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Project

## Quick Run

```bash
.\run_project.bat
```

## Manual Run

```bash
python main.py
```

## Run with Custom Parameters

```bash
python main.py --dataset path\to\RML2016.10a_dict_optimized.pkl --output-dir outputs --sample-limit 30000 --open-dashboard
```

---

# Open Jupyter Notebook

```bash
jupyter notebook Mbu_finalyear_project.ipynb
```

---

# Generated Outputs

The project automatically saves:

* RF Signal Plot
* Severity Distribution Plot
* Confusion Matrix
* Feature Importance Graph
* Interactive HTML Dashboard

Output directory:

```bash
outputs/
```

The dashboard provides a frontend-style visualization interface explaining all generated results and evaluation metrics.

---

# Research Motivation

The project addresses growing concerns regarding RF coexistence between 5G communication systems and aviation radio altimeters.

Traditional analytical methods explain nonlinear interference behavior but lack adaptive and real-time predictive capabilities.

This work introduces a hybrid RF + ML framework capable of:

* Detecting nonlinear gain compression
* Predicting altitude-error escalation
* Identifying severe interference conditions
* Supporting onboard radio altimeter safety monitoring

---

# Future Improvements

* Integration of real flight-test datasets
* Deep learning–based temporal RF analysis
* Real-time streaming interference detection
* Edge deployment optimization
* FPGA or SDR-based implementation

---

# Conference & Academic Work

This project was developed as a final year major project under the Department of Computer Science and Engineering, Mohan Babu University.

The work was also presented as part of an IEEE conference-oriented academic presentation on RF interference prediction and aviation safety monitoring.

---

# IEEE Conference Presentation

This project was presented as part of an IEEE conference-oriented academic research presentation focused on RF coexistence, aviation safety, and machine learning–based interference detection in radio altimeters.

The presentation covered:

* 5G-induced nonlinear interference in aviation radio altimeters
* RF signal analysis and spectral feature engineering
* Random Forest and SVM-based interference classification
* Height-error prediction and safety monitoring
* Real-time interference detection logic

The work demonstrates the application of machine learning in RF engineering and aviation communication safety systems.


# Author

Yashwanth Varakuti

Department of Computer Science and Engineering
Mohan Babu University

GitHub: https://github.com/Yashwanthvarakuti

---

# License

This project is intended for academic, research, and educational purposes.
