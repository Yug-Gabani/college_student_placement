import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score   
from preprocess import preprocess_data

X, y, label_encoder = preprocess_data(
    "college_student_placement_dataset.csv",
    is_training=True
)

joblib.dump(list(X.columns), "feature_names.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "placement_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Training complete. Files created successfully.")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
