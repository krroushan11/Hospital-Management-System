import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Hospital Management Dashboard", layout="wide")

st.title("🏥 Hospital Management System Dashboard")

# Load datasets
patients = pd.read_csv("Dataset/patients.csv")
doctors = pd.read_csv("Dataset/doctors.csv")
appointments = pd.read_csv("Dataset/appointments.csv")
billing = pd.read_csv("Dataset/billing.csv")
treatments = pd.read_csv("Dataset/treatments.csv")
doctor_encoder = joblib.load("Model/doctor_encoder.pkl")
patient_encoder = joblib.load("Model/patient_encoder.pkl")
appointment_encoder = joblib.load("Model/appointment_encoder.pkl")
# Load model
model = joblib.load("Model/model.pkl")

# Sidebar
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Select Option",
    ["Patients", "Doctors", "Appointments", "Analytics", "Prediction", "Revenue"]
)


# Patients
if option == "Patients":
    st.subheader("Patients Data")
    st.dataframe(patients)

# Doctors
if option == "Doctors":
    st.subheader("Doctors Data")
    st.dataframe(doctors)

# Appointments
if option == "Appointments":
    st.subheader("Appointments Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Appointment Status Distribution")
        st.bar_chart(appointments['status'].value_counts())

    with col2:
        st.write("Appointments per Doctor")
        st.bar_chart(appointments['doctor_id'].value_counts())
# Analytics
if option == "Analytics":
    st.subheader("Hospital Analytics Dashboard")
     # KPI Dashboard
    colA, colB, colC = st.columns(3)

    colA.metric("Total Patients", len(patients))
    colB.metric("Total Doctors", len(doctors))
    colC.metric("Total Revenue", billing['amount'].sum())

    
        # Appointment Status Pie Chart
    st.subheader("Appointment Status Distribution")

    import matplotlib.pyplot as plt

    status_counts = appointments['status'].value_counts()

    fig, ax = plt.subplots()
    ax.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%')
    ax.set_title("Appointment Status")

    st.pyplot(fig)
        # Doctor Workload Heatmap
    st.subheader("Doctor Workload Heatmap")

    import seaborn as sns

    workload = appointments.groupby(['doctor_id','status']).size().unstack()

    fig, ax = plt.subplots()
    sns.heatmap(workload, cmap="coolwarm", annot=True)

    st.pyplot(fig)
    col1, col2 = st.columns(2)
    

    # 1 Patients by Gender
    with col1:
        st.write("Patients by Gender")
        st.bar_chart(patients['gender'].value_counts())

    # 2 Doctors by Specialization
    with col2:
        st.write("Doctors by Specialization")
        st.bar_chart(doctors['specialization'].value_counts())

    col3, col4 = st.columns(2)

    # 3 Appointment Status
    with col3:
        st.write("Appointment Status Distribution")
        st.bar_chart(appointments['status'].value_counts())

    # 4 Appointments per Doctor
    with col4:
        st.write("Appointments per Doctor")
        st.bar_chart(appointments['doctor_id'].value_counts())

    col5, col6 = st.columns(2)

    # 5 Revenue Distribution
    with col5:
        st.write("Billing Amount Distribution")
        st.line_chart(billing['amount'])

    # 6 Revenue by Patient
    with col6:
        revenue_patient = billing.groupby('patient_id')['amount'].sum()
        st.write("Revenue by Patient")
        st.bar_chart(revenue_patient)

    col7, col8 = st.columns(2)

    # 7 Treatments Frequency
    with col7:
        st.write("Most Common Treatments")
        st.bar_chart(treatments['treatment_id'].value_counts())

    # 8 Treatment Cost
    with col8:
        st.write("Treatment Cost Analysis")
        st.bar_chart(treatments['cost'])

    col9, col10 = st.columns(2)

    # 9 Doctor Workload
    with col9:
        workload = appointments.groupby('doctor_id').size()
        st.write("Doctor Workload")
        st.bar_chart(workload)

    # 10 Patients by City
    with col10:
        if 'city' in patients.columns:
            st.write("Patients by City")
            st.bar_chart(patients['city'].value_counts())

    col11, col12 = st.columns(2)

    # 11 Billing Trend
    with col11:
        st.write("Billing Trend")
        st.line_chart(billing['amount'])

    # 12 Appointment Trend
    with col12:
        if 'date' in appointments.columns:
            appointments['date'] = pd.to_datetime(appointments['date'])
            trend = appointments.groupby(appointments['date'].dt.date).size()
            st.write("Daily Appointment Trend")
            st.line_chart(trend)

# Revenue
if option == "Revenue":
    st.subheader("Hospital Revenue")

    total_revenue = billing['amount'].sum()

    st.metric("Total Revenue", total_revenue)

    st.bar_chart(billing['amount'])

# Prediction
if option == "Prediction":

    st.subheader("Predict Appointment Status")

    doctor_id = st.selectbox(
        "Select Doctor ID",
        appointments['doctor_id'].unique()
    )

    patient_id = st.selectbox(
        "Select Patient ID",
        appointments['patient_id'].unique()
    )

    appointment_id = st.selectbox(
        "Select Appointment ID",
        appointments['appointment_id'].unique()
    )

if st.button("Predict"):

    doctor_val = doctor_encoder.transform([doctor_id])[0]
    patient_val = patient_encoder.transform([patient_id])[0]
    appointment_val = appointment_encoder.transform([appointment_id])[0]

    input_data = [[doctor_val, patient_val, appointment_val]]

    result = model.predict(input_data)

    status_map = {0:"Scheduled",1:"Completed",2:"Cancelled"}

    st.success(f"Predicted Status: {status_map[result[0]]}")
