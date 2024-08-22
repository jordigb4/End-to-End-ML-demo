import shap
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from pyarrow import parquet as pq
from catboost import CatBoostClassifier, Pool
import joblib

MODEL_PATH = "models/catboost_model.cbm"
DATA_PATH = "data/heart_risk.parquet"

st.set_page_config(page_title='Heart Risk Project')


@st.cache_resource
def load_data():
    data = pd.read_parquet(DATA_PATH)
    return data


def load_x_y(file_path):
    data = joblib.load(file_path)
    data.reset_index(drop=True, inplace=True)
    return data


def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model

def calculate_shap(model, X_train, X_test):
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_cat_train = explainer.shap_values(X_train)
    shap_values_cat_test = explainer.shap_values(X_test)
    return explainer, shap_values_cat_train, shap_values_cat_test


def plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, customer_id, X_test, X_train):
    # Visualize SHAP values for a specific customer
    fig, ax_2 = plt.subplots(figsize=(6,6), dpi=200)
    shap.decision_plot(explainer.expected_value, shap_values_cat_test[customer_id], X_test.loc[customer_id], link="logit")
    st.pyplot(fig)
    plt.close()


def display_shap_summary(shap_values_cat_train, X_train):
    # Create the plot summarizing the SHAP values
    shap.summary_plot(shap_values_cat_train, X_train, plot_type="bar", plot_size=(12,12))
    summary_fig, _ = plt.gcf(), plt.gca()
    st.pyplot(summary_fig)
    plt.close()


def display_shap_waterfall_plot(explainer, expected_value, shap_values, feature_names, max_display=20):
    # Create SHAP waterfall drawing
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    shap.plots._waterfall.waterfall_legacy(expected_value, shap_values, feature_names=feature_names, max_display=max_display, show=False)
    st.pyplot(fig)
    plt.close()


def summary(model, data, X_train, X_test):
    # Calculate SHAP values
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)

    # Summarize and visualize SHAP values
    display_shap_summary(shap_values_cat_train, X_train)


def plot_shap(model, data, index, X_train, X_test):
    # Calculate SHAP values
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)

    # Visualize SHAP values
    plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, index, X_test, X_train)

    # Waterfall

    display_shap_waterfall_plot(explainer, explainer.expected_value, shap_values_cat_test[index],
                                feature_names=X_test.columns, max_display=20)


st.title("Heart Risk Prediction Project")


def main():
    model = load_model()
    data = load_data()

    X_train = load_x_y("data/X_train.pkl")
    X_test = load_x_y("data/X_test.pkl")
    y_train = load_x_y("data/y_train.pkl")
    y_test = load_x_y("data/y_test.pkl")

    # Radio buttons for options
    election = st.radio("Make Your Choice:",
                        ("Feature Importance", "User-based SHAP", "Calculate the probability of Heart Risk"))
    available_customer_ids = list(range(X_test.shape[0]))

    # If User-based SHAP option is selected
    if election == "User-based SHAP":
        # Customer ID text input
        customer_id = st.selectbox("Choose the Customer", available_customer_ids)
        st.write(f'Customer {customer_id}: Actual value for the Customer Heart Risk : {y_test.iloc[customer_id]}')
        y_pred = model.predict(X_test)
        st.write(
            f"Customer {customer_id}: CatBoost Model's prediction for the Customer Heart Risk : {y_pred[customer_id]}")
        plot_shap(model, data, customer_id, X_train=X_train, X_test=X_test)

    # If Feature Importance is selected
    elif election == "Feature Importance":
        summary(model, data, X_train=X_train, X_test=X_test)

    # If Calculate RISK Probability option is selected
    elif election == "Calculate the probability of Heart Risk":

        # Selectbox for categorical variables

        sex = st.selectbox("Sex:", ("Male", "Female"))
        diabetes = st.selectbox("Diabetes (0: No, 1: Yes):", (0, 1))
        fam_hist = st.selectbox("Family History (0: No, 1: Yes):", (0, 1))
        smoking = st.selectbox("Smoking (0: No, 1: Yes):", (0, 1))
        obesity = st.selectbox("Obesity (0: No, 1: Yes):", (0, 1))
        alcohol = st.selectbox("Alcohol Consumption (0: No, 1: Yes):", (0, 1))
        diet = st.selectbox("Diet:", ("Healthy", "Unhealthy", "Balanced"))
        heart_problems = st.selectbox("Previous Heart Problems (0: No, 1: Yes):", (0, 1))
        medication = st.selectbox("Medication Use (0: No, 1: Yes):", (0, 1))
        stress = st.selectbox("Stress Level (0-10):", sorted(X_train['continent'].unique()))
        country = st.selectbox("Country:", sorted(X_train['country'].unique()))
        continent = st.selectbox("Continent:", sorted(X_train['continent'].unique()))
        hemisphere = st.selectbox("Hemisphere:", ("Northern Hemisphere", "Southern Hemisphere"))

        # Number input for numerical variables
        age = st.number_input("Age:", min_value=0, max_value=120, step=1)
        cholesterol = st.number_input("Cholesterol:", min_value=0, max_value=500, step=1)
        heart_rate = st.number_input("Heart Rate:", min_value=0, max_value=200, step=1)
        exersice = st.number_input("Exercise Hours Per Week:", min_value=0.0, max_value=168.0, step=0.1)
        sedentary_hours = st.number_input("Sedentary Hours Per Day:", min_value=0.0, max_value=24.0, step=0.1)
        income = st.number_input("Income:", min_value=0, max_value=1000000, step=1)
        bmi = st.number_input("BMI:", min_value=0.0, max_value=100.0, step=0.1)
        triglycerides = st.number_input("Triglycerides:", min_value=0, max_value=1000, step=1)
        physical_activity_days = st.number_input("Physical Activity Days Per Week:", min_value=0, max_value=7, step=1)
        sleep = st.number_input("Sleep Hours Per Day:", min_value=0, max_value=24, step=1)
        systolic = st.number_input("Systolic Blood Pressure:", min_value=0, max_value=300, step=1)
        diastolic = st.number_input("Diastolic Blood Pressure:", min_value=0, max_value=200, step=1)


        # Confirmation button
        confirmation_button = st.button("Confirm")

        # When the confirmation button is clicked
        if confirmation_button:
            # Convert user-entered data into a data frame
            new_customer_data = pd.DataFrame({
                'age': [age],
                'sex': [sex],
                'cholesterol': [cholesterol],
                'heart_rate': [heart_rate],
                'diabetes': [diabetes],
                'fam_hist': [fam_hist],
                'smoking': [smoking],
                'obesity': [obesity],
                'alcohol': [alcohol],
                'exersice': [exersice],
                'diet': [diet],
                'heart_problems': [heart_problems],
                'medication': [medication],
                'stress': [stress],
                'sedentary_hours': [sedentary_hours],
                'income': [income],
                'BMI': [bmi],
                'triglycerides': [triglycerides],
                'physical_activity_days': [physical_activity_days],
                'sleep': [sleep],
                'country': [country],
                'continent': [continent],
                'hemisphere': [hemisphere],
                'systolic': [systolic],
                'diastolic': [diastolic]
            })

            # Predict RISK probability using the model
            risk_probability = model.predict_proba(new_customer_data)[:, 1]

            # Format risk probability
            formatted_risk_probability = "{:.2%}".format(risk_probability.item())

            big_text = f"<h1>Heart attack Probability: {formatted_risk_probability}</h1>"
            st.markdown(big_text, unsafe_allow_html=True)
            st.write(new_customer_data.to_dict())


if __name__ == "__main__":
    main()























