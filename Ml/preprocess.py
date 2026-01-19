import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(csv_path, is_training=True):
    df = pd.read_csv(csv_path)

    df["Internship_Experience"] = df["Internship_Experience"].map({
        "Yes": 1,
        "No": 0
    })

    if is_training:
        label_encoder = LabelEncoder()
        df["Placement"] = label_encoder.fit_transform(df["Placement"])

        X = df.drop(columns=["Placement", "College_ID"])
        y = df["Placement"]

        return X, y, label_encoder
    else:
        X = df.drop(columns=["College_ID"])
        return X
