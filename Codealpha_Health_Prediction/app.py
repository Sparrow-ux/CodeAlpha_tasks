from sklearn import model_selection
import streamlit as st
import joblib
import pandas as pd


heart = joblib.load("heart_disease_model.pkl")
diabetes = joblib.load("diabetes_model.pkl")
Dscaler = joblib.load("scaler_for_diabeties.pkl")

st.title("Healthcare Diagnosis Prediction System")

tab1, tab2, tab3 = st.tabs(["Heart Disease", "Diabetes","About Us"])

with tab1:

    st.header("Heart Disease Prediction")

    heart_age = st.number_input("Age",min_value = 1,key="heart_age")
    sex = st.selectbox("Gender",["Female", "Male"])
    sex = 0 if sex == "Female" else 1
    cp = st.number_input("Chest pain type (0-3)",min_value = 0,max_value = 3)
    tresbps = st.number_input("Resting blood pressure")
    chol = st.number_input("Cholestrol")
    fbs = st.number_input("Fasting blood sugar (0 or 1)",min_value = 0,max_value = 1)
    restecg = st.number_input("Resting ECG (0-2)",min_value = 0,max_value = 2)
    thalach = st.number_input("Maximum Heart rate")
    exang = st.selectbox("Exercise Induced Angina",["No", "Yes"])
    exang = 0 if exang == "No" else 1
    oldpeak = st.number_input("ST depression induced by exercise relative to rest")
    slope = st.number_input("Slope of the peak exercise ST segment (0-2)",min_value = 0,max_value = 2)
    ca = st.number_input("Number of major vessels (0-3)",min_value = 0,max_value = 3)
    thal = st.number_input("Thalium stress test result (0-2)",min_value = 0,max_value = 3)


    if(st.button("Predict Heart Disease")):
        features = pd.DataFrame([[heart_age,sex,cp,tresbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]],columns = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"])
        
        predictions = heart.predict_proba(features)
        st.write("Probability of Heart Disease :",abs((predictions[0][1]*100)-100),"%")


with tab2:

    st.header("Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose",key="diabetes_glucose")
    blood_pressure = st.number_input("Blood Pressure")
    skin_thickness = st.number_input("Skin Thickness in mm")
    insulin = st.number_input("Insulin")
    bmi = st.slider("BMI",min_value=10.0,max_value=60.0,value=25.0)
    dpf = st.slider("Diabetes Pedigree Function",min_value = 0.0,max_value = 2.42,value = 0.21)
    diabetes_age = st.number_input("Age",min_value=1,key="diabetes_age")

    if st.button("Predict Diabetes"):

        diabetes_data = pd.DataFrame([[pregnancies, glucose, blood_pressure,skin_thickness, insulin, bmi,dpf, diabetes_age]],columns=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"])

        diabetes_data = Dscaler.transform(diabetes_data)

        prediction = diabetes.predict(diabetes_data)[0]

        probabilities = diabetes.predict_proba(diabetes_data)[0]
        diabetes_probability = probabilities[0] * 100
        st.write("Probability of Diabetes :",abs(diabetes_probability-100),"%")



with tab3:
    st.header("About us")
    st.write("This application predicts the likelihood of Heart Disease and Diabetes using trained Machine Learning models.")
    st.write("Made by : Deepak Raaja H J")
    st.write("")
    st.write("")
    st.write("Models used for predictions:")
    st.write("Heart Disease Prediction: XGBoost Classifier")
    st.write("Model Metrics:")
    st.write("Accuracy of Heart disease model : 84 %")
    st.write("Precision of Heart disease model : 84.37 %")
    st.write("Recall score of Heart disease model : 84.37 %")
    st.write("F1_Score of Heart disease model : 84.37 %")
    st.write("Training accuracy of Heart disease model : 88.42975206611571 %")
    st.write("Testing accuracy of Heart disease model : 83.60655737704919 %")
    st.write("")
    st.write("")
    st.write("Diabetes Prediction: SVM Classifier")
    st.write("Model Metrics:")
    st.write("Accuracy of Diabetes model : 71.48 %")
    st.write("Precision of Diabetes model : 57.78 %")
    st.write("Recall score of Diabetes model : 74.54 %")
    st.write("F1_Score of Diabetes model : 65.78 %")
    st.write("Training accuracy of Diabetes model : 89.95 %")
    st.write("Testing accuracy of Diabetes model : 77.08 %")
    st.write("")
    st.write("")
    st.write("Note: These models are trained on historical data and should be used as a predictive tool only. For accurate diagnosis and treatment, please consult a healthcare professional.")