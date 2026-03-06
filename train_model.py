import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

appointments = pd.read_csv("Dataset/appointments.csv")

doctor_encoder = LabelEncoder()
patient_encoder = LabelEncoder()
appointment_encoder = LabelEncoder()

appointments['doctor_id'] = doctor_encoder.fit_transform(appointments['doctor_id'])
appointments['patient_id'] = patient_encoder.fit_transform(appointments['patient_id'])
appointments['appointment_id'] = appointment_encoder.fit_transform(appointments['appointment_id'])

appointments['status'] = appointments['status'].astype('category').cat.codes

X = appointments[['doctor_id','patient_id','appointment_id']]
y = appointments['status']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)

joblib.dump(model,"Model/model.pkl")
joblib.dump(doctor_encoder,"Model/doctor_encoder.pkl")
joblib.dump(patient_encoder,"Model/patient_encoder.pkl")
joblib.dump(appointment_encoder,"Model/appointment_encoder.pkl")

print("Model trained and encoders saved")