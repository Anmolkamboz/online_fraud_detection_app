import streamlit as st
import numpy as np
import joblib
# for aws = '/home/ubuntu/online_fraud_detection/my_model.joblib'
model = joblib.load("C:\\projects\\online_fraud_detection_app\\random_forest_model.joblib") 
type_to_number = {
    'PAYMENT': 3,
    'TRANSFER': 4,
    'CASH_OUT': 1,
    'DEBIT': 2,
    'CASH_IN': 5
}
def main():
    st.title("Online Fraud Detector")
    st.write("Enter Values")



    feature1 = st.selectbox("Type", list(type_to_number.keys()))
    feature1 = type_to_number.get(feature1)
    feature2 = st.number_input("Amount")
    feature3 = st.number_input("Old Balance Orig")
    feature4 = st.number_input("New Balance Orig")

    if st.button("Predict"):
        features = np.array([[feature1, feature2, feature3, feature4]])
        prediction = model.predict(features)
        st.success(f"Predicted class: {prediction[0]}")

if __name__ == "__main__":
    main()
