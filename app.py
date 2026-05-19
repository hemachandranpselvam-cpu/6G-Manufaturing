import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="6G Smart Factory Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🏭 Manufacturing Process Health and Operational Efficiency Dashboard")
st.markdown("### 6G-Enabled Smart Factory Analytics")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Thales_Group_Manufacturing.csv")
    except FileNotFoundError:
        st.error(
            "Thales_Group_Manufacturing.csv not found. "
            "Place the CSV file in the same folder as app.py"
        )
        st.stop()

    df.columns = df.columns.str.strip()

    # Create timestamp
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    elif "Date" in df.columns and "Time" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce",
        )

    elif "Date" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Date"], errors="coerce")

    else:
        st.error("No valid date/timestamp columns found.")
        st.stop()

    df = df.dropna(subset=["Timestamp"]).copy()

    # Time Features
    df["Hour"] = df["Timestamp"].dt.hour
    df["Day"] = df["Timestamp"].dt.day
    df["Month"] = df["Timestamp"].dt.month
    df["Weekday"] = df["Timestamp"].dt.day_name()

    return df


df = load_data()

# ---------------------------------------------------
# REQUIRED COLUMNS CHECK
# ---------------------------------------------------

required_columns = [
    "Machine_ID",
    "Operation_Mode",
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score",
    "Error_Rate_%",
    "Efficiency_Status",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------

st.sidebar.header("🔍 Dashboard Filters")

machine_filter = st.sidebar.multiselect(
    "Select Machine",
    options=sorted(df["Machine_ID"].unique()),
    default=sorted(df["Machine_ID"].unique()),
)

mode_filter = st.sidebar.multiselect(
    "Select Operation Mode",
    options=sorted(df["Operation_Mode"].unique()),
    default=sorted(df["Operation_Mode"].unique()),
)

efficiency_filter = st.sidebar.multiselect(
    "Select Efficiency Status",
    options=sorted(df["Efficiency_Status"].unique()),
    default=sorted(df["Efficiency_Status"].unique()),
)

start_date = st.sidebar.date_input(
    "Start Date",
    df["Timestamp"].min().date()
)

end_date = st.sidebar.date_input(
    "End Date",
    df["Timestamp"].max().date()
)

# ---------------------------------------------------
# FILTER DATA
# ---------------------------------------------------

filtered_df = df[
    (df["Machine_ID"].isin(machine_filter))
    & (df["Operation_Mode"].isin(mode_filter))
    & (df["Efficiency_Status"].isin(efficiency_filter))
    & (df["Timestamp"].dt.date >= start_date)
    & (df["Timestamp"].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("No records found for selected filters.")
    st.stop()

# ---------------------------------------------------
# MACHINE HEALTH INDEX
# ---------------------------------------------------

health_features = filtered_df[
    [
        "Temperature_C",
        "Vibration_Hz",
        "Power_Consumption_kW",
        "Predictive_Maintenance_Score",
    ]
].fillna(0)

scaler = MinMaxScaler()

scaled = scaler.fit_transform(health_features)

filtered_df["Machine_Health_Index"] = (
    scaled[:, 0] * 0.35
    + scaled[:, 1] * 0.30
    + scaled[:, 2] * 0.20
    + scaled[:, 3] * 0.15
)

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🏭 Factory Overview",
        "🩺 Machine Health",
        "📈 Production & Quality",
        "⚙️ Efficiency Diagnostics",
    ]
)

# ===================================================
# TAB 1 — FACTORY OVERVIEW
# ===================================================

with tab1:

    st.subheader("📌 Factory Health Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Machines",
            filtered_df["Machine_ID"].nunique()
        )

    with col2:
        st.metric(
            "Avg Temperature",
            round(filtered_df["Temperature_C"].mean(), 2)
        )

    with col3:
        st.metric(
            "Avg Production Speed",
            round(filtered_df["Production_Speed_units_per_hr"].mean(), 2)
        )

    with col4:
        st.metric(
            "Avg Defect Rate",
            round(filtered_df["Quality_Control_Defect_Rate_%"].mean(), 2)
        )

    with col5:
        st.metric(
            "Avg Error Rate",
            round(filtered_df["Error_Rate_%"].mean(), 2)
        )

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:

        fig_eff = px.pie(
            filtered_df,
            names="Efficiency_Status",
            hole=0.4,
            title="Efficiency Status Distribution",
        )

        st.plotly_chart(fig_eff, use_container_width=True)

    with col_b:

        sensor_avg = filtered_df[
            [
                "Temperature_C",
                "Vibration_Hz",
                "Power_Consumption_kW",
                "Network_Latency_ms",
            ]
        ].mean().reset_index()

        sensor_avg.columns = ["Metric", "Average"]

        fig_sensor = px.bar(
            sensor_avg,
            x="Metric",
            y="Average",
            color="Metric",
            title="Average Sensor Metrics",
        )

        st.plotly_chart(fig_sensor, use_container_width=True)

    st.markdown("---")

    fig_prod_trend = px.line(
        filtered_df,
        x="Timestamp",
        y="Production_Speed_units_per_hr",
        color="Machine_ID",
        title="Production Speed Trend",
    )

    st.plotly_chart(fig_prod_trend, use_container_width=True)

# ===================================================
# TAB 2 — MACHINE HEALTH
# ===================================================

with tab2:

    st.subheader("🩺 Machine Health Dashboard")

    machine_list = sorted(filtered_df["Machine_ID"].unique())

    selected_machine = st.selectbox(
         "Select Machine",
          machine_list,
    )

    machine_df = filtered_df[
        filtered_df["Machine_ID"] == selected_machine
    ]

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Health Index",
            round(machine_df["Machine_Health_Index"].mean(), 2)
        )

    with col2:
        st.metric(
            "Avg Temperature",
            round(machine_df["Temperature_C"].mean(), 2)
        )

    with col3:
        st.metric(
            "Avg Vibration",
            round(machine_df["Vibration_Hz"].mean(), 2)
        )

    with col4:
        st.metric(
            "Maintenance Score",
            round(machine_df["Predictive_Maintenance_Score"].mean(), 2)
        )

    st.markdown("---")

    fig_temp = px.line(
        machine_df,
        x="Timestamp",
        y="Temperature_C",
        title="Temperature Trend",
    )

    st.plotly_chart(fig_temp, use_container_width=True)

    fig_vibration = px.line(
        machine_df,
        x="Timestamp",
        y="Vibration_Hz",
        title="Vibration Trend",
    )

    st.plotly_chart(fig_vibration, use_container_width=True)

    fig_power = px.line(
        machine_df,
        x="Timestamp",
        y="Power_Consumption_kW",
        title="Power Consumption Trend",
    )

    st.plotly_chart(fig_power, use_container_width=True)

    fig_maintenance = px.area(
        machine_df,
        x="Timestamp",
        y="Predictive_Maintenance_Score",
        title="Predictive Maintenance Score Trend",
    )

    st.plotly_chart(fig_maintenance, use_container_width=True)

# ===================================================
# TAB 3 — PRODUCTION & QUALITY
# ===================================================

with tab3:

    st.subheader("📈 Production & Quality Panel")

    col1, col2 = st.columns(2)

    with col1:

        fig_prod_defect = px.scatter(
            filtered_df,
            x="Production_Speed_units_per_hr",
            y="Quality_Control_Defect_Rate_%",
            color="Operation_Mode",
            size="Error_Rate_%",
            hover_data=["Machine_ID"],
            title="Production Speed vs Defect Rate",
        )

        st.plotly_chart(fig_prod_defect, use_container_width=True)

    with col2:

        fig_error = px.line(
            filtered_df,
            x="Timestamp",
            y="Error_Rate_%",
            color="Machine_ID",
            title="Operational Error Rate Over Time",
        )

        st.plotly_chart(fig_error, use_container_width=True)

    st.markdown("---")

    defect_machine = (
        filtered_df.groupby("Machine_ID")[
            "Quality_Control_Defect_Rate_%"
        ]
        .mean()
        .reset_index()
    )

    fig_defect_machine = px.bar(
        defect_machine,
        x="Machine_ID",
        y="Quality_Control_Defect_Rate_%",
        color="Quality_Control_Defect_Rate_%",
        title="Average Defect Rate by Machine",
    )

    st.plotly_chart(fig_defect_machine, use_container_width=True)

    st.markdown("---")

    fig_temp_defect = px.scatter(
        filtered_df,
        x="Temperature_C",
        y="Quality_Control_Defect_Rate_%",
        color="Operation_Mode",
        title="Temperature vs Defect Rate",
    )

    st.plotly_chart(fig_temp_defect, use_container_width=True)

# ===================================================
# TAB 4 — EFFICIENCY DIAGNOSTICS
# ===================================================

with tab4:

    st.subheader("⚙️ Efficiency Diagnostics View")

    col1, col2 = st.columns(2)

    with col1:

        fig_efficiency = px.histogram(
            filtered_df,
            x="Efficiency_Status",
            color="Efficiency_Status",
            title="Efficiency Status Breakdown",
        )

        st.plotly_chart(fig_efficiency, use_container_width=True)

    with col2:

        fig_mode = px.box(
            filtered_df,
            x="Operation_Mode",
            y="Production_Speed_units_per_hr",
            color="Operation_Mode",
            title="Operation Mode vs Production Speed",
        )

        st.plotly_chart(fig_mode, use_container_width=True)

    st.markdown("---")

    efficiency_machine = (
        filtered_df.groupby("Machine_ID")[
            "Production_Speed_units_per_hr"
        ]
        .mean()
        .reset_index()
    )

    fig_eff_machine = px.bar(
        efficiency_machine,
        x="Machine_ID",
        y="Production_Speed_units_per_hr",
        color="Production_Speed_units_per_hr",
        title="Machine Efficiency Comparison",
    )

    st.plotly_chart(fig_eff_machine, use_container_width=True)

    st.markdown("---")

    heatmap_df = filtered_df.pivot_table(
        values="Production_Speed_units_per_hr",
        index="Machine_ID",
        columns="Efficiency_Status",
        aggfunc="mean",
    )

    fig_heat = px.imshow(
        heatmap_df,
        text_auto=True,
        aspect="auto",
        title="Machine Efficiency Heatmap",
    )

    st.plotly_chart(fig_heat, use_container_width=True)

# ===================================================
# NETWORK DIAGNOSTICS
# ===================================================

st.markdown("---")

st.subheader("📡 6G Network Diagnostics")

col_net1, col_net2 = st.columns(2)

with col_net1:

    fig_latency = px.histogram(
        filtered_df,
        x="Network_Latency_ms",
        nbins=30,
        title="Network Latency Distribution",
    )

    st.plotly_chart(fig_latency, use_container_width=True)

with col_net2:

    fig_packet = px.scatter(
        filtered_df,
        x="Packet_Loss_%",
        y="Error_Rate_%",
        color="Operation_Mode",
        title="Packet Loss vs Error Rate",
    )

    st.plotly_chart(fig_packet, use_container_width=True)

# ===================================================
# MACHINE SUMMARY TABLE
# ===================================================

st.markdown("---")

st.subheader("📋 Machine Performance Summary")

summary = filtered_df.groupby("Machine_ID").agg(
    {
        "Temperature_C": "mean",
        "Vibration_Hz": "mean",
        "Power_Consumption_kW": "mean",
        "Production_Speed_units_per_hr": "mean",
        "Quality_Control_Defect_Rate_%": "mean",
        "Error_Rate_%": "mean",
        "Machine_Health_Index": "mean",
    }
).round(2)

st.dataframe(summary, use_container_width=True)

# ===================================================
# EXECUTIVE INSIGHTS
# ===================================================

st.markdown("---")

st.subheader("🧠 Executive Insights")

st.info(
    """
    • High-load operation modes show elevated defect rates.
    
    • Machines with higher vibration trends demonstrate increased operational errors.
    
    • Temperature spikes strongly correlate with quality degradation.
    
    • Some machines consume higher power without proportional production gains.
    
    • Increased packet loss and latency may affect operational consistency.
    """
)

# ===================================================
# DOWNLOAD SECTION
# ===================================================

st.markdown("---")

st.subheader("⬇️ Download Filtered Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="filtered_factory_data.csv",
    mime="text/csv",
)

# ===================================================
# FOOTER
# ===================================================

st.markdown("---")

st.markdown("### ✅ Smart Factory Operational Analytics System")

st.markdown(
    "Developed using Streamlit + Plotly + Industrial IoT Analytics"
)
