import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("car_price_prediction_.csv")

df.columns = df.columns.str.strip().str.lower()

df = df.drop("car id", axis=1)


label_cols = ["brand", "fuel type", "transmission", "condition", "model"]

le = LabelEncoder()

for col in label_cols:
    df[col] = le.fit_transform(df[col])

# print(df.isnull().sum())


X = df.drop("price", axis=1)  
y = df["price"]  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

