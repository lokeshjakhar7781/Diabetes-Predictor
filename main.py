import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipline(num_attribs,cat_attribs):
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,num_attribs),
        ("cat",cat_pipeline,cat_attribs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    data = pd.read_csv("diabetes_prediction_dataset.csv")
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in split.split(data,data["HbA1c_level"]):
        test_dataset = data.loc[test_index].to_csv("input.csv",index=False)
        df = data.loc[train_index]
    
    df_labels = df["diabetes"].copy()
    df_features = df.drop("diabetes",axis=1)
    df_num_attribs = df_features.select_dtypes(include=[np.number]).columns.tolist()
    df_cat_attribs = df_features.select_dtypes(exclude=[np.number]).columns.tolist()
    pipeline = build_pipline(df_num_attribs,df_cat_attribs)
    df_prepared = pipeline.fit_transform(df_features)
    model = RandomForestClassifier(random_state=42)
    model.fit(df_prepared,df_labels)
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Model is trained and saved")

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    input_data = pd.read_csv("input.csv")
    transformed_data = pipeline.transform(input_data)
    predictions = model.predict(transformed_data)
    input_data["diabetes"] = predictions
    input_data["diabetes"].to_csv("output.csv",index=False)
    print("Inference is complete and output saved to output.csv")