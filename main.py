from pathlib import Path
import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


DEFAULT_DATASETS = (
    "RML2016.10a_dict.pkl",
    "RML2016.10a_dict_optimized.pkl",
)
DEFAULT_OUTPUT_DIR = "outputs"
RANDOM_STATE = 42


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def resolve_dataset_path(dataset_path):
    if not dataset_path.exists():
        for default_dataset in DEFAULT_DATASETS:
            candidate = Path(default_dataset)
            if candidate.exists():
                return candidate

        expected_names = " or ".join(DEFAULT_DATASETS)
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Place {expected_names} in the project root, or pass its path "
            "with --dataset."
        )

    return dataset_path


def load_dataset(dataset_path, sample_limit):
    dataset_path = resolve_dataset_path(dataset_path)

    print(f"Loading Dataset: {dataset_path}")
    with dataset_path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    signals = []
    snr_values = []
    for (_, snr), iq_signals in data.items():
        for signal in iq_signals:
            signals.append(signal)
            snr_values.append(snr)

    x = np.array(signals)
    snr_array = np.array(snr_values)

    if sample_limit:
        x = x[:sample_limit]
        snr_array = snr_array[:sample_limit]

    print("Dataset Shape:", x.shape)
    return x, snr_array


def extract_features(signals):
    features = []

    for signal in signals:
        i_signal = signal[0]
        q_signal = signal[1]
        magnitude = np.sqrt(i_signal**2 + q_signal**2)

        features.append(
            [
                np.sqrt(np.mean(magnitude**2)),
                np.var(magnitude),
                skew(magnitude),
                kurtosis(magnitude),
            ]
        )

    return np.array(features)


def build_interference_dataframe(features, snr_values, random_state):
    rng = np.random.default_rng(random_state)
    n = len(features)

    f_5g = rng.uniform(3500, 4500, n)
    p_5g = -40 + (snr_values + 20) * (35 / 40)
    b_5g = rng.uniform(20, 100, n)

    freq_peak = np.exp(-((f_5g - 4300) / 200) ** 2)
    power_soft = sigmoid((p_5g + 22) / 2)
    power_hard = sigmoid((p_5g + 15) / 1.2)

    gamma = 0.02 + 0.5 * power_soft + 0.25 * power_hard + 0.15 * freq_peak
    gamma = np.clip(gamma, 0, 0.7)

    sir = 35 - 15 * power_soft - 10 * freq_peak - 10 * gamma
    sir = np.clip(sir, 0, 40)

    rms = features[:, 0]
    if_power = 20 * np.log10(rms + 1e-6)
    echo_power = if_power - (10 + 15 * gamma)

    height_error = (
        0.005
        + 0.03 * power_soft
        + 0.02 * power_hard
        + 0.02 * freq_peak
        + 0.04 * gamma
        + rng.normal(0, 0.002, n)
    )
    height_error = np.clip(height_error, 0, 0.2)

    labels = np.zeros(n)
    labels[height_error > 0.02] = 1
    labels[height_error > 0.05] = 2

    df = pd.DataFrame(
        features,
        columns=["RMS", "Variance", "Skewness", "Kurtosis"],
    )
    df["f_5G"] = f_5g
    df["P_5G"] = p_5g
    df["B_5G"] = b_5g
    df["Gain_compression_ratio"] = gamma
    df["SIR"] = sir
    df["IF_output_power"] = if_power
    df["Echo_power"] = echo_power
    df["height_error_ratio"] = height_error
    df["severity_class"] = labels.astype(int)

    return df


def save_visualizations(signals, df, model, x_test, y_test, feature_columns, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = signals[0]
    plt.figure(figsize=(8, 4))
    plt.plot(sample[0], label="I Signal")
    plt.plot(sample[1], label="Q Signal")
    plt.title("Raw RF IQ Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "signal_plot.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(x="severity_class", data=df)
    plt.title("Interference Severity Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "severity_distribution.png")
    plt.close()

    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    pd.Series(model.feature_importances_, index=feature_columns).sort_values().plot(
        kind="barh",
        figsize=(8, 6),
    )
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png")
    plt.close()


def run_analysis(dataset_path, output_dir, sample_limit):
    signals, snr_values = load_dataset(dataset_path, sample_limit)

    features = extract_features(signals)
    print("Feature Shape:", features.shape)

    df = build_interference_dataframe(features, snr_values, RANDOM_STATE)
    print(df.head())

    x_data = df.drop(["severity_class", "height_error_ratio"], axis=1)
    y = df["severity_class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)

    print("Training Random Forest Model...")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("\nAccuracy:", accuracy_score(y_test, predictions))
    print("F1 Score:", f1_score(y_test, predictions, average="macro"))
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    probabilities = model.predict_proba(x_test)
    roc_auc = roc_auc_score(y_bin, probabilities, multi_class="ovr")
    print("ROC-AUC:", roc_auc)

    save_visualizations(
        signals,
        df,
        model,
        x_test,
        y_test,
        x_data.columns,
        output_dir,
    )

    print(f"\nPlots saved to: {output_dir}")
    print("Project Executed Successfully!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze 5G interference severity from RF IQ signal data."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASETS[0],
        help="Path to RML2016.10a_dict.pkl or RML2016.10a_dict_optimized.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=30000,
        help="Maximum number of samples to use. Set 0 to use all samples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sample_limit = args.sample_limit or None
    run_analysis(Path(args.dataset), Path(args.output_dir), sample_limit)


if __name__ == "__main__":
    main()
