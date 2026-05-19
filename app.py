import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(
    page_title="6G Smart Factory Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Manufacturing Process Health and Operational Efficiency Dashboard")
st.markdown("### 6G-Enabled Smart Factory Analytics")


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Thales_Group_Manufacturing.csv")
    except FileNotFoundError:
        st.error(
            "Could not find Thales_Group_Manufacturing.csv. "
            "Place it in the same folder/repository as app.py."
        )
        st.stop()

    df.columns = df.columns.str.strip()

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
        st.error(f"No Date/Timestamp column found. Columns are: {df.columns.tolist()}")
        st.stop()

    df = df.dropna(subset=["Timestamp"]).copy()
    df["Hour"] = df["Timestamp"].dt.hour
    df["Day"] = df["Timestamp"].dt.day
    df["Month"] = df["Timestamp"].dt.month
    return df


def require_columns(df, columns):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()


df = load_data()

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
require_columns(df, required_columns)

st.sidebar.header("Dashboard Filters")

machine_options = sorted(df["Machine_ID"].dropna().unique())
mode_options = sorted(df["Operation_Mode"].dropna().unique())

machine_filter = st.sidebar.multiselect(
    "Select Machine",
    options=machine_options,
    default=machine_options,
)

mode_filter = st.sidebar.multiselect(
    "Select Operation Mode",
    options=mode_options,
    default=mode_options,
)

start_date = st.sidebar.date_input("Start Date", df["Timestamp"].min().date())
end_date = st.sidebar.date_input("End Date", df["Timestamp"].max().date())

filtered_df = df[
    (df["Machine_ID"].isin(machine_filter))
    & (df["Operation_Mode"].isin(mode_filter))
    & (df["Timestamp"].dt.date >= start_date)
    & (df["Timestamp"].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

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

st.subheader("Factory Health Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Machines", filtered_df["Machine_ID"].nunique())

with col2:
    st.metric("Avg Temperature", round(filtered_df["Temperature_C"].mean(), 2))

with col3:
    st.metric(
        "Avg Production Speed",
        round(filtered_df["Production_Speed_units_per_hr"].mean(), 2),
    )

with col4:
    st.metric(
        "Avg Defect Rate",
        round(filtered_df["Quality_Control_Defect_Rate_%"].mean(), 2),
    )

with col5:
    st.metric("Avg Error Rate", round(filtered_df["Error_Rate_%"].mean(), 2))

st.subheader("Efficiency Status Distribution")
fig_eff = px.pie(
    filtered_df,
    names="Efficiency_Status",
    title="Efficiency Status Breakdown",
)
st.plotly_chart(fig_eff, use_container_width=True)

st.subheader("Machine Health Dashboard")
health_machine = (
    filtered_df.groupby("Machine_ID")
    [["Temperature_C", "Vibration_Hz", "Power_Consumption_kW", "Machine_Health_Index"]]
    .mean()
    .reset_index()
)

fig_health = px.bar(
    health_machine,
    x="Machine_ID",
    y="Machine_Health_Index",
    color="Machine_Health_Index",
    title="Machine Health Index by Machine",
)
st.plotly_chart(fig_health, use_container_width=True)

st.subheader("Temperature Trend Analysis")
fig_temp = px.scatter(
    filtered_df,
    x="Temperature_C",
    y="Quality_Control_Defect_Rate_%",
    color="Operation_Mode",
    size="Error_Rate_%",
    hover_data=["Machine_ID"],
    title="Temperature vs Defect Rate",
)
st.plotly_chart(fig_temp, use_container_width=True)

st.subheader("Error Frequency Analysis")
fig_error = px.line(
    filtered_df,
    x="Timestamp",
    y="Error_Rate_%",
    color="Machine_ID",
    title="Operational Error Rate Over Time",
)
st.plotly_chart(fig_error, use_container_width=True)

st.subheader("6G Network Diagnostics")
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

st.subheader("Power Consumption vs Efficiency")
fig_power = px.scatter(
    filtered_df,
    x="Power_Consumption_kW",
    y="Production_Speed_units_per_hr",
    color="Efficiency_Status",
    size="Machine_Health_Index",
    hover_data=["Machine_ID"],
    title="Power Consumption vs Production Speed",
)
st.plotly_chart(fig_power, use_container_width=True)

st.subheader("Machine Performance Summary")
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

st.subheader("Download Filtered Data")
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="filtered_factory_data.csv",
    mime="text/csv",
)

st.markdown("---")
st.markdown("### Smart Factory Operational Analytics System")
st.markdown("Developed using Streamlit + Plotly + Industrial IoT Analytics")
