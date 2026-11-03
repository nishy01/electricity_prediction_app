import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from train import FEATURE_COLS, MODEL_PATH, SUMMARY_PATH, train_and_save

# 1. Page Configuration
st.set_page_config(
    page_title="Household Electricity Prediction",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Household Electricity Consumption Prediction System")
st.write(
    "Enter your household electrical readings and time of day below to predict "
    "**Global Active Power (kW)** using our trained Machine Learning model."
)


# 2. Load Pre-Trained Model & Summary Data (or train on first run)
@st.cache_resource
def get_model_and_summary():
    if os.path.exists(MODEL_PATH) and os.path.exists(SUMMARY_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(SUMMARY_PATH, "rb") as f:
                summary = pickle.load(f)
            _ = model.predict(
                pd.DataFrame([[240.0, 0.1, 0.0, 0.0, 0.0, 12, 0, 1]], columns=FEATURE_COLS)
            )
            return model, summary
        except Exception:
            return train_and_save()
    else:
        return train_and_save()


with st.spinner("Loading Machine Learning model and dataset summary... (First run takes ~20s to train)"):
    model, summary = get_model_and_summary()

medians = summary["medians"]
metrics = summary["metrics"]

# ==========================================
# SECTION 1: MANUAL INPUT & LIVE PREDICTION
# ==========================================
st.header("🔮 Step 1: Enter Manual Inputs to Predict Power Output")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔌 Electrical Metrics")
    voltage = st.number_input(
        "Voltage (Volts)",
        min_value=220.0,
        max_value=255.0,
        value=float(medians["Voltage"]),
        step=0.5,
        help="Minute-averaged voltage in Volts (typical range: 230V - 245V)",
    )
    reactive_power = st.number_input(
        "Global Reactive Power (kW)",
        min_value=0.0,
        max_value=1.5,
        value=float(medians["Global_reactive_power"]),
        step=0.02,
        help="Minute-averaged reactive power in kilowatts (typical range: 0.0 - 0.5 kW)",
    )

with col2:
    st.subheader("🏠 Appliance Sub-Metering (Wh)")
    sub_1 = st.number_input(
        "Sub-metering 1: Kitchen (Wh)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        help="Active energy for kitchen (dishwasher, oven, microwave) in Watt-hours",
    )
    sub_2 = st.number_input(
        "Sub-metering 2: Laundry Room (Wh)",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=1.0,
        help="Active energy for laundry (washing machine, dryer, fridge, light) in Watt-hours",
    )
    sub_3 = st.number_input(
        "Sub-metering 3: Water Heater & AC (Wh)",
        min_value=0.0,
        max_value=50.0,
        value=17.0,
        step=1.0,
        help="Active energy for electric water-heater and air-conditioner in Watt-hours",
    )

with col3:
    st.subheader("🕒 Time & Calendar")
    hour = st.slider("Hour of Day (0 = Midnight, 23 = 11 PM)", 0, 23, 18)
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_choice = st.selectbox("Day of Week", day_names, index=0)
    day_of_week = day_names.index(day_choice)

    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    month_choice = st.selectbox("Month", month_names, index=11)
    month = month_names.index(month_choice) + 1

if st.button("⚡ Predict Global Active Power", type="primary", use_container_width=True):
    input_df = pd.DataFrame(
        [[voltage, reactive_power, sub_1, sub_2, sub_3, hour, day_of_week, month]],
        columns=FEATURE_COLS,
    )
    pred_kw = max(0.0, float(model.predict(input_df)[0]))
    avg_kw = summary["avg_active_power"]
    diff_pct = ((pred_kw - avg_kw) / avg_kw) * 100

    st.success("### 🎯 Prediction Result")
    r1, r2, r3 = st.columns(3)
    r1.metric(
        label="Predicted Global Active Power",
        value=f"{pred_kw:.3f} kW",
        delta=f"{diff_pct:+.1f}% vs. Dataset Average ({avg_kw:.2f} kW)",
        delta_color="inverse",
    )
    r2.metric(
        label="Estimated Hourly Consumption",
        value=f"{pred_kw:.2f} kWh",
    )
    r3.metric(
        label="Estimated Current (Global Intensity)",
        value=f"{(pred_kw * 1000 / voltage):.1f} Amps",
    )

st.divider()

# ==========================================
# SECTION 2: MODEL PERFORMANCE & VISUALIZATIONS
# ==========================================
st.header("📊 Step 2: Model Accuracy & Dataset Insights")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Test R² Score", f"{metrics['R2']:.3f}")
m2.metric("Test MAE", f"{metrics['MAE']:.3f} kW")
m3.metric("Test RMSE", f"{metrics['RMSE']:.3f} kW")
m4.metric("Total Dataset Rows", f"{summary['total_rows']:,}")

c1, c2 = st.columns(2)
with c1:
    if st.checkbox("Show Raw Data Preview (First 20 Rows)"):
        st.dataframe(summary["head_df"])
with c2:
    if st.checkbox("Show Dataset Statistical Summary"):
        st.dataframe(summary["describe_df"])

tab1, tab2, tab3 = st.tabs([
    "🎯 Actual vs. Predicted (Test Sample)",
    "📈 Daily Average Active Power",
    "🔥 Feature Correlation Heatmap",
])

with tab1:
    fig_pred, ax_pred = plt.subplots(figsize=(14, 5))
    ax_pred.plot(
        summary["sample_index"],
        summary["sample_actual"],
        label="Actual Power (kW)",
        color="black",
        alpha=0.6,
    )
    ax_pred.plot(
        summary["sample_index"],
        summary["sample_pred"],
        label="Predicted Power (kW)",
        color="tab:red",
        linestyle="--",
        alpha=0.85,
    )
    ax_pred.set_title("Actual vs. Predicted Global Active Power (First 24 Hours of Test Set)")
    ax_pred.set_xlabel("Time")
    ax_pred.set_ylabel("Global Active Power (kW)")
    ax_pred.legend()
    ax_pred.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_pred)

with tab2:
    daily_power = summary["daily_active_power"]
    fig_daily, ax_daily = plt.subplots(figsize=(14, 5))
    ax_daily.plot(daily_power.index, daily_power.values, color="tab:red", alpha=0.8)
    ax_daily.set_title("Daily Average Global Active Power (kW)")
    ax_daily.set_xlabel("Date")
    ax_daily.set_ylabel("Active Power (kW)")
    ax_daily.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_daily)

with tab3:
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        summary["corr_matrix"],
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax_corr,
    )
    ax_corr.set_title("Correlation Matrix of Household Metrics")
    st.pyplot(fig_corr)
