import joblib
import pandas as pd
from preprocess import preprocess_data

model = joblib.load("placement_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

print("\nEnter student details for placement prediction\n")

IQ = int(input("🧠IQ: "))
CGPA = float(input("CGPA: "))
Prev_Sem_Result = float(input("Previous Semester Result: "))
Academic_Performance = int(input("📚Academic Performance (1-10): "))
Communication_Skills = int(input("Communication Skills (1-10): "))
Extra_Curricular_Score = int(input("Extra Curricular Score (1-10): "))
Internship_Experience = input("Internship Experience (Yes/No): ")

data = {
    "College_ID": [1],
    "IQ": [IQ],
    "CGPA": [CGPA],
    "Prev_Sem_Result": [Prev_Sem_Result],
    "Academic_Performance": [Academic_Performance],
    "Communication_Skills": [Communication_Skills],
    "Extra_Curricular_Score": [Extra_Curricular_Score],
    "Internship_Experience": [Internship_Experience]
}

df = pd.DataFrame(data)

df.to_csv("input.csv", index=False)

X_input = preprocess_data("input.csv", is_training=False)

for col in feature_names:
    if col not in X_input.columns:
        X_input[col] = 0

X_input = X_input[feature_names]

prediction = model.predict(X_input)
result = label_encoder.inverse_transform(prediction)

print("\n🎯 Placement Prediction:", result[0])
