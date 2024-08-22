import pandas as pd
from catboost import CatBoostClassifier

# Load the trained model
MODEL_PATH = "models/catboost_model.cbm"
model = CatBoostClassifier()
model.load_model(MODEL_PATH)


def predict_risk(user_input):
    try:
        # Prepare data for prediction
        user_data = pd.DataFrame([user_input])

        # Make prediction
        prediction = model.predict_proba(user_data)[:, 1][0]

        return {"Heart Risk Probability": float(prediction)}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Ask user for input
    customerID = "6464-UIAEA"
    print("Please enter the following information:")

    # Input prompts for the data
    age = int(input("Age: ").strip())
    sex = input("Sex (Male/Female): ").strip()
    cholesterol = int(input("Cholesterol: ").strip())
    heart_rate = int(input("Heart Rate: ").strip())
    diabetes = int(input("Diabetes (0/1): ").strip())
    fam_hist = int(input("Family History (0/1): ").strip())
    smoking = int(input("Smoking (0/1): ").strip())
    obesity = int(input("Obesity (0/1): ").strip())
    alcohol = int(input("Alcohol Consumption (0/1): ").strip())
    exersice = float(input("Exercise Hours Per Week: ").strip())
    diet = input("Diet (Healthy/Unhealthy/Balanced): ").strip()
    heart_problems = int(input("Previous Heart Problems (0/1): ").strip())
    medication = int(input("Medication Use (0/1): ").strip())
    stress = int(input("Stress Level (0-10): ").strip())
    sedentary_hours = float(input("Sedentary Hours Per Day: ").strip())
    income = int(input("Income: ").strip())
    bmi = float(input("BMI: ").strip())
    triglycerides = int(input("Triglycerides: ").strip())
    physical_activity_days = int(input("Physical Activity Days Per Week: ").strip())
    sleep = int(input("Sleep Hours Per Day: ").strip())
    country = input("Country: ").strip()
    continent = input("Continent: ").strip()
    hemisphere = input("Hemisphere (Northern Hemisphere/Southern Hemisphere): ").strip()
    systolic = int(input("Systolic Blood Pressure: ").strip())
    diastolic = int(input("Diastolic Blood Pressure: ").strip())

    # Create a dictionary with the inputs
    data = {
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
    }

    # Convert the dictionary to a DataFrame
    new_data = pd.DataFrame(data)

    # Predict heart risk probability using the model
    risk_probability = model.predict_proba(new_data)[:, 1]

    # Format heart risk probability
    formatted_heartrisk_probability = "{:.2%}".format(risk_probability.item())

    print(f"Heart Risk Probability: {formatted_heartrisk_probability}")