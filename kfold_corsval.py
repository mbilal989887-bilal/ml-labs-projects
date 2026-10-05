import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("Used_car_prices_in_Pakistan.csv")

df.columns = df.columns.str.strip().str.lower()

df['version'].fillna(df['version'].mode()[0], inplace=True)
df.drop('registered city', axis=1, inplace=True)

df['price'] = pd.to_numeric(df['price'], errors='coerce')
df.dropna(subset=['price'], inplace=True)

label_cols = ["make", "model", "version", "assembly", "transmission"]

le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop("price", axis=1)
y = df["price"]

model = LinearRegression()

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
mae_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
mse_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')

print("K-Fold Cross Validation Results (k=5)")
print("------------------------------------")
print("Average R² Score:", r2_scores.mean())
print("Average MAE:", -mae_scores.mean())
print("Average MSE:", -mse_scores.mean())
print("Average RMSE:", np.sqrt(-mse_scores.mean()))
