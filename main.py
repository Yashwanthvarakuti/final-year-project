from pathlib import Path
import argparse
import webbrowser
import pickle

import matplotlib

matplotlib.use("Agg")
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


def save_dashboard(output_dir, metrics, class_counts):
    dashboard_path = output_dir / "dashboard.html"
    class_rows = "\n".join(
        f"<tr><td>Class {label}</td><td>{count}</td></tr>"
        for label, count in class_counts.items()
    )

    dashboard_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>5G Interference Analysis Dashboard</title>
  <style>
    :root {{
      --ink: #19202a;
      --muted: #5b6472;
      --line: #d8dee8;
      --panel: #ffffff;
      --page: #f5f7fb;
      --accent: #0f766e;
      --accent-2: #b45309;
      --accent-3: #334155;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--page);
      color: var(--ink);
    }}

    header {{
      background: #111827;
      color: #ffffff;
      padding: 32px 24px;
      border-bottom: 5px solid var(--accent);
    }}

    header .wrap,
    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
    }}

    h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 44px);
      line-height: 1.1;
      letter-spacing: 0;
    }}

    h2 {{
      margin: 0 0 10px;
      font-size: 22px;
      letter-spacing: 0;
    }}

    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }}

    header p {{
      color: #d7dde8;
      max-width: 780px;
    }}

    main {{
      padding: 24px 0 36px;
    }}

    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }}

    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}

    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 8px;
    }}

    .metric strong {{
      display: block;
      font-size: 30px;
      line-height: 1;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}

    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}

    section img {{
      display: block;
      width: 100%;
      height: auto;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}

    .copy {{
      padding: 18px;
    }}

    .tag {{
      display: inline-block;
      margin-bottom: 10px;
      color: #ffffff;
      background: var(--accent-3);
      border-radius: 4px;
      padding: 5px 8px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}

    .signal .tag {{
      background: var(--accent);
    }}

    .severity .tag {{
      background: var(--accent-2);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 15px;
    }}

    td {{
      border-top: 1px solid var(--line);
      padding: 10px 0;
    }}

    td:last-child {{
      text-align: right;
      font-weight: 700;
    }}

    .note {{
      margin-top: 22px;
      background: #ffffff;
      border: 1px solid var(--line);
      border-left: 5px solid var(--accent);
      border-radius: 8px;
      padding: 18px;
    }}

    @media (max-width: 760px) {{
      .metrics,
      .grid {{
        grid-template-columns: 1fr;
      }}

      header {{
        padding: 26px 18px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>5G Interference Analysis Dashboard</h1>
      <p>This dashboard summarizes the machine learning output for RF IQ signal data. It shows the signal pattern, class distribution, model accuracy, confusion matrix, and which engineered features influenced the Random Forest model most.</p>
    </div>
  </header>

  <main>
    <div class="metrics" aria-label="Model metrics">
      <div class="metric">
        <span>Accuracy</span>
        <strong>{metrics["accuracy"]:.3f}</strong>
      </div>
      <div class="metric">
        <span>Macro F1 Score</span>
        <strong>{metrics["f1"]:.3f}</strong>
      </div>
      <div class="metric">
        <span>ROC-AUC</span>
        <strong>{metrics["roc_auc"]:.3f}</strong>
      </div>
    </div>

    <div class="grid">
      <section class="signal">
        <img src="signal_plot.png" alt="Raw RF IQ signal line chart">
        <div class="copy">
          <span class="tag">Signal View</span>
          <h2>Raw RF IQ Signal</h2>
          <p>This image shows one sample from the dataset. The I and Q curves represent the two parts of the radio signal over time. These wave patterns are converted into numerical features before training the model.</p>
        </div>
      </section>

      <section class="severity">
        <img src="severity_distribution.png" alt="Interference severity class distribution">
        <div class="copy">
          <span class="tag">Class Balance</span>
          <h2>Interference Severity Distribution</h2>
          <p>This chart shows how many samples fall into each severity class. Class 0 means lower interference, class 1 means medium interference, and class 2 means higher interference.</p>
          <table aria-label="Severity class counts">
            {class_rows}
          </table>
        </div>
      </section>

      <section>
        <img src="confusion_matrix.png" alt="Random Forest confusion matrix">
        <div class="copy">
          <span class="tag">Model Check</span>
          <h2>Confusion Matrix</h2>
          <p>This image compares the true severity class with the class predicted by the model. Darker diagonal cells mean the model is correctly identifying most samples in that class.</p>
        </div>
      </section>

      <section>
        <img src="feature_importance.png" alt="Random Forest feature importance bar chart">
        <div class="copy">
          <span class="tag">Feature Impact</span>
          <h2>Feature Importance</h2>
          <p>This chart ranks the input features used by the Random Forest. Longer bars mean that feature had more influence on the model's interference severity prediction.</p>
        </div>
      </section>
    </div>

    <div class="note">
      <h2>Result Summary</h2>
      <p>The model achieved strong performance on the test set. The dashboard images help explain both the signal data and the model behavior in a visual format suitable for project presentation.</p>
    </div>
  </main>
</body>
</html>
"""
    dashboard_path.write_text(dashboard_html, encoding="utf-8")
    return dashboard_path


def run_analysis(dataset_path, output_dir, sample_limit, open_dashboard=False):
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

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    print("\nAccuracy:", accuracy)
    print("F1 Score:", f1)
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

    dashboard_path = save_dashboard(
        output_dir,
        {"accuracy": accuracy, "f1": f1, "roc_auc": roc_auc},
        df["severity_class"].value_counts().sort_index().to_dict(),
    )

    print(f"\nPlots saved to: {output_dir}")
    print(f"Dashboard saved to: {dashboard_path}")
    if open_dashboard:
        webbrowser.open(dashboard_path.resolve().as_uri())
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
    parser.add_argument(
        "--open-dashboard",
        action="store_true",
        help="Open the generated dashboard in the default browser.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sample_limit = args.sample_limit or None
    run_analysis(
        Path(args.dataset),
        Path(args.output_dir),
        sample_limit,
        open_dashboard=args.open_dashboard,
    )


if __name__ == "__main__":
    main()
