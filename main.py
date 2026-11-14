"""
Fragile AI for Ocean Forecasting
Baseline experiment:
- Target: next-day Sea Surface Temperature (SST)
- Inputs: previous 7 consecutive daily observations at the same location
- Model: scikit-learn MLPRegressor
- Split: chronological 80:20 by unique dates
- Robustness tests: Gaussian noise, random missingness, sensor/feature dropout

Run:
    python main.py
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "fragile_ai_ocean_forecasting_dataset.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

LOOKBACK = 7
TRAIN_FRACTION = 0.80
RANDOM_STATE = 42

# SST is the prediction target. Temperature is excluded because it duplicates SST.
FEATURES = ["SST", "Salinity", "Wind_Speed", "Latitude", "Longitude"]
PERTURBABLE_FEATURES = ["SST", "Salinity", "Wind_Speed"]


def load_and_validate(path: Path) -> pd.DataFrame:
    """Load the CSV and validate required columns and basic data types."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place fragile_ai_ocean_forecasting_dataset.csv inside the data folder."
        )

    df = pd.read_csv(path)
    required = ["Date", "Latitude", "Longitude", "SST", "Salinity", "Temperature", "Wind_Speed"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        bad = int(df["Date"].isna().sum())
        raise ValueError(f"Found {bad} rows with invalid dates. Inspect the source CSV.")

    numeric_cols = ["Latitude", "Longitude", "SST", "Salinity", "Temperature", "Wind_Speed"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Required coordinates and target must exist. SST is the target.
    df = df.dropna(subset=["Latitude", "Longitude", "SST"]).copy()

    # Remove exact duplicate records, then order each location chronologically.
    df = df.drop_duplicates()
    df = df.sort_values(["Latitude", "Longitude", "Date"]).reset_index(drop=True)

    # The uploaded file has Temperature identical to SST wherever both exist.
    # It is therefore excluded from model inputs to avoid redundant information.
    if df["Temperature"].notna().any():
        same = np.isclose(
            df.loc[df["Temperature"].notna(), "Temperature"],
            df.loc[df["Temperature"].notna(), "SST"],
            equal_nan=False
        ).all()
        if same:
            print("Check: Temperature duplicates SST; excluding Temperature from features.")
        else:
            print("Note: Temperature differs from SST in some rows; still excluded for this SST experiment.")

    return df


def chronological_split_dates(df: pd.DataFrame):
    """Choose a global cutoff using unique dates, not randomly shuffled rows."""
    dates = np.array(sorted(df["Date"].drop_duplicates()))
    if len(dates) < 20:
        raise ValueError("Not enough unique dates for a reliable chronological split.")

    split_index = int(len(dates) * TRAIN_FRACTION)
    split_index = min(max(split_index, 1), len(dates) - 1)
    cutoff = pd.Timestamp(dates[split_index - 1])
    return cutoff


def fit_training_medians(df: pd.DataFrame, cutoff: pd.Timestamp):
    """Fit missing-value replacements using training-period rows only."""
    train_rows = df[df["Date"] <= cutoff]
    medians = train_rows[["Salinity", "Wind_Speed"]].median()

    # If a whole training feature is missing, fail rather than silently inventing a value.
    if medians.isna().any():
        raise ValueError("A feature has no usable training values; review the dataset.")
    return medians


def make_windows(df: pd.DataFrame, cutoff: pd.Timestamp, medians: pd.Series):
    """
    Create samples separately within each geographic location.
    Every input window must contain 7 consecutive calendar days.
    The target is SST on the immediately following day.
    Test targets are after the cutoff; their preceding history may include
    training-period observations, as it would in real forecasting.
    """
    work = df.copy()
    work["Salinity"] = work["Salinity"].fillna(medians["Salinity"])
    work["Wind_Speed"] = work["Wind_Speed"].fillna(medians["Wind_Speed"])

    # A row must have one record per location/date for an unambiguous daily sequence.
    duplicated_keys = work.duplicated(["Latitude", "Longitude", "Date"]).sum()
    if duplicated_keys:
        raise ValueError(
            f"Found {duplicated_keys} duplicate location/date records. "
            "Resolve these before generating daily windows."
        )

    X_train, y_train, X_test, y_test = [], [], [], []
    train_dates, test_dates = [], []

    for (lat, lon), group in work.groupby(["Latitude", "Longitude"], sort=False):
        group = group.sort_values("Date").reset_index(drop=True)
        dates = group["Date"].to_numpy(dtype="datetime64[ns]")
        values = group[FEATURES].to_numpy(dtype=float)
        targets = group["SST"].to_numpy(dtype=float)

        # Window = 7 days of input; next row/day is the target.
        for end in range(LOOKBACK, len(group)):
            input_start = end - LOOKBACK
            input_dates = dates[input_start:end]
            target_date = pd.Timestamp(dates[end])

            # Require exact daily continuity for all 7 inputs and the target day.
            expected = np.arange(
                input_dates[0],
                dates[end] + np.timedelta64(1, "D"),
                np.timedelta64(1, "D")
            )
            observed = dates[input_start:end + 1]
            if len(expected) != LOOKBACK + 1 or not np.array_equal(observed, expected):
                continue

            sample = values[input_start:end].reshape(-1)
            target = targets[end]

            if target_date <= cutoff:
                X_train.append(sample)
                y_train.append(target)
                train_dates.append(target_date)
            else:
                X_test.append(sample)
                y_test.append(target)
                test_dates.append(target_date)

    if not X_train or not X_test:
        raise ValueError(
            "Not enough consecutive daily windows in train or test period. "
            "Check the dataset's date coverage and location continuity."
        )

    return (
        np.asarray(X_train), np.asarray(y_train),
        np.asarray(X_test), np.asarray(y_test),
        np.asarray(train_dates), np.asarray(test_dates),
        work
    )


def build_model():
    """Baseline MLP regression pipeline; scaler is fitted on training windows only."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=RANDOM_STATE
        ))
    ])


def metric_values(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred))
    }


def perturb_gaussian(X, train_feature_std, level, rng):
    """Add Gaussian noise to SST, salinity and wind-speed input positions."""
    Xp = X.copy().reshape(-1, LOOKBACK, len(FEATURES))
    for feature in PERTURBABLE_FEATURES:
        j = FEATURES.index(feature)
        Xp[:, :, j] += rng.normal(
            0.0, level * train_feature_std[feature],
            size=Xp[:, :, j].shape
        )
    return Xp.reshape(len(X), -1)


def perturb_missing(X, train_medians, ratio, rng):
    """Replace randomly selected cells in perturbable channels with train medians."""
    Xp = X.copy().reshape(-1, LOOKBACK, len(FEATURES))
    for feature in PERTURBABLE_FEATURES:
        j = FEATURES.index(feature)
        mask = rng.random(Xp[:, :, j].shape) < ratio
        replacement = train_medians[feature] if feature in train_medians else 0.0
        Xp[:, :, j][mask] = replacement
    return Xp.reshape(len(X), -1)


def perturb_sensor_dropout(X, train_medians):
    """Simulate loss of the salinity and wind-speed channels for all test windows."""
    Xp = X.copy().reshape(-1, LOOKBACK, len(FEATURES))
    for feature in ["Salinity", "Wind_Speed"]:
        j = FEATURES.index(feature)
        Xp[:, :, j] = train_medians[feature]
    return Xp.reshape(len(X), -1)


def save_plots(y_true, clean_pred, results_df):
    # Actual vs predicted
    plt.figure(figsize=(8, 6))
    sample_n = min(1000, len(y_true))
    plt.scatter(y_true[:sample_n], clean_pred[:sample_n], alpha=0.45)
    low = min(float(np.min(y_true[:sample_n])), float(np.min(clean_pred[:sample_n])))
    high = max(float(np.max(y_true[:sample_n])), float(np.max(clean_pred[:sample_n])))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual SST")
    plt.ylabel("Predicted SST")
    plt.title("Actual vs Predicted SST (Clean Test Data)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close()

    # Fragility curve: RMSE against perturbation level
    plt.figure(figsize=(8, 6))
    for name, subset in results_df.groupby("Experiment"):
        if name.startswith("Gaussian noise") or name.startswith("Random missingness"):
            plt.plot(subset["Level"], subset["RMSE"], marker="o", label=name)
    plt.xlabel("Perturbation level")
    plt.ylabel("RMSE")
    plt.title("Forecasting Fragility under Input Perturbations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fragility_curve.png", dpi=180)
    plt.close()

    # Residual histogram
    residuals = y_true - clean_pred
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=40)
    plt.xlabel("Actual SST - Predicted SST")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution (Clean Test Data)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_distribution.png", dpi=180)
    plt.close()


def main():
    print("\n--- 1. Loading and validating dataset ---")
    df = load_and_validate(DATA_PATH)
    print(f"Rows after validation: {len(df):,}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Unique locations: {df[['Latitude', 'Longitude']].drop_duplicates().shape[0]}")

    print("\n--- 2. Chronological split and training-only imputation ---")
    cutoff = chronological_split_dates(df)
    medians = fit_training_medians(df, cutoff)
    print(f"Training target dates: up to {cutoff.date()}")
    print(f"Training medians: {medians.to_dict()}")

    print("\n--- 3. Creating 7-day consecutive windows ---")
    X_train, y_train, X_test, y_test, train_dates, test_dates, processed_df = make_windows(
        df, cutoff, medians
    )
    print(f"Training windows: {len(X_train):,}")
    print(f"Testing windows: {len(X_test):,}")
    print(f"Input features per sample: {X_train.shape[1]} ({LOOKBACK} days x {len(FEATURES)} features)")

    # Save a cleaned copy for transparency. Temperature is dropped because it duplicates SST.
    cleaned = processed_df.drop(columns=["Temperature"])
    cleaned.to_csv(OUTPUT_DIR / "cleaned_ocean_data.csv", index=False)

    print("\n--- 4. Training baseline MLPRegressor ---")
    model = build_model()
    model.fit(X_train, y_train)
    clean_pred = model.predict(X_test)
    clean_metrics = metric_values(y_test, clean_pred)
    print("Clean test metrics:", clean_metrics)

    joblib.dump(model, MODEL_DIR / "sst_mlp_model.joblib")
    joblib.dump(
        {"features": FEATURES, "lookback": LOOKBACK, "cutoff": cutoff,
         "training_medians": medians.to_dict()},
        MODEL_DIR / "preprocessing_metadata.joblib"
    )

    print("\n--- 5. Robustness experiments ---")
    # Compute feature standard deviations using training-period rows only.
    train_rows = processed_df[processed_df["Date"] <= cutoff]
    train_feature_std = train_rows[FEATURES].std()
    train_feature_std = train_feature_std.fillna(0.0)

    rng = np.random.default_rng(RANDOM_STATE)
    experiment_rows = [{
        "Experiment": "Clean baseline",
        "Level": 0.0,
        **clean_metrics,
        "MAE_degradation_percent": 0.0,
        "RMSE_degradation_percent": 0.0
    }]

    for level in [0.1, 0.25, 0.5]:
        Xp = perturb_gaussian(X_test, train_feature_std, level, rng)
        pred = model.predict(Xp)
        m = metric_values(y_test, pred)
        experiment_rows.append({
            "Experiment": "Gaussian noise",
            "Level": level,
            **m,
            "MAE_degradation_percent": 100 * (m["MAE"] - clean_metrics["MAE"]) / max(clean_metrics["MAE"], 1e-8),
            "RMSE_degradation_percent": 100 * (m["RMSE"] - clean_metrics["RMSE"]) / max(clean_metrics["RMSE"], 1e-8)
        })

    for ratio in [0.1, 0.25, 0.5]:
        Xp = perturb_missing(X_test, medians.to_dict(), ratio, rng)
        pred = model.predict(Xp)
        m = metric_values(y_test, pred)
        experiment_rows.append({
            "Experiment": "Random missingness",
            "Level": ratio,
            **m,
            "MAE_degradation_percent": 100 * (m["MAE"] - clean_metrics["MAE"]) / max(clean_metrics["MAE"], 1e-8),
            "RMSE_degradation_percent": 100 * (m["RMSE"] - clean_metrics["RMSE"]) / max(clean_metrics["RMSE"], 1e-8)
        })

    Xp = perturb_sensor_dropout(X_test, medians.to_dict())
    pred = model.predict(Xp)
    m = metric_values(y_test, pred)
    experiment_rows.append({
        "Experiment": "Sensor dropout (Salinity + Wind_Speed)",
        "Level": 1.0,
        **m,
        "MAE_degradation_percent": 100 * (m["MAE"] - clean_metrics["MAE"]) / max(clean_metrics["MAE"], 1e-8),
        "RMSE_degradation_percent": 100 * (m["RMSE"] - clean_metrics["RMSE"]) / max(clean_metrics["RMSE"], 1e-8)
    })

    results = pd.DataFrame(experiment_rows)
    results.to_csv(OUTPUT_DIR / "robustness_results.csv", index=False)
    pd.DataFrame({
        "Target_Date": test_dates,
        "Actual_SST": y_test,
        "Predicted_SST_Clean": clean_pred
    }).to_csv(OUTPUT_DIR / "clean_test_predictions.csv", index=False)

    print(results.to_string(index=False))

    print("\n--- 6. Saving visualizations ---")
    save_plots(y_test, clean_pred, results)
    print("\nCompleted.")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print(f"Trained model saved in: {MODEL_DIR}")
    print("Review robustness_results.csv before reporting any performance claims.")


if __name__ == "__main__":
    main()
