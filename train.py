import os
import zipfile
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "individual+household+electric+power+consumption.zip")
TXT_PATH = os.path.join(BASE_DIR, "household_power_consumption.txt")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SUMMARY_PATH = os.path.join(BASE_DIR, "summary_data.pkl")

FEATURE_COLS = [
    "Voltage",
    "Global_reactive_power",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "Hour",
    "Day_of_week",
    "Month",
]
TARGET_COL = "Global_active_power"


def load_and_clean_data():
    print("1. Loading raw dataset...")
    if os.path.exists(TXT_PATH):
        df = pd.read_csv(TXT_PATH, sep=";", low_memory=False, na_values="?")
    elif os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            data_file = [f for f in z.namelist() if f.endswith(".txt") or f.endswith(".csv")][0]
            with z.open(data_file) as f:
                df = pd.read_csv(f, sep=";", low_memory=False, na_values="?")
    else:
        raise FileNotFoundError("Dataset file not found!")

    print(f"   Loaded {len(df):,} rows. Cleaning and parsing timestamps...")
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    df_clean = df.dropna(subset=["Datetime"]).copy()
    df_clean.set_index("Datetime", inplace=True)
    df_clean.drop(columns=["Date", "Time"], inplace=True)

    # Fill missing values using time-series forward fill
    df_clean = df_clean.ffill().bfill()

    # Create time features
    df_clean["Hour"] = df_clean.index.hour
    df_clean["Day_of_week"] = df_clean.index.dayofweek
    df_clean["Month"] = df_clean.index.month

    return df_clean


def train_and_save():
    df = load_and_clean_data()

    print("2. Splitting dataset chronologically (85% Train, 15% Test)...")
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    print("3. Training Gradient Boosting Regressor model...")
    model = HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)

    print("4. Evaluating model on test set...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    print(f"   Test R2: {r2:.4f} | MAE: {mae:.4f} kW | RMSE: {rmse:.4f} kW")

    print("5. Saving trained model and dashboard summary data...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    numeric_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
    ]
    summary_data = {
        "metrics": {"MSE": float(mse), "RMSE": rmse, "MAE": mae, "R2": r2},
        "total_rows": len(df),
        "avg_active_power": float(df[TARGET_COL].mean()),
        "medians": df[FEATURE_COLS].median().to_dict(),
        "head_df": df.head(20),
        "describe_df": df[numeric_cols].describe().round(3),
        "daily_active_power": df["Global_active_power"].resample("D").mean(),
        "corr_matrix": df[numeric_cols].corr().round(3),
        "sample_actual": y_test.iloc[:1440].values,
        "sample_pred": y_pred[:1440],
        "sample_index": y_test.iloc[:1440].index,
    }
    with open(SUMMARY_PATH, "wb") as f:
        pickle.dump(summary_data, f)

    print("Done! Model saved to model.pkl and summary saved to summary_data.pkl.")
    return model, summary_data


if __name__ == "__main__":
    train_and_save()
