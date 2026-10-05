import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("Used_car_prices_in_Pakistan.csv")

df.columns = df.columns.str.strip().str.lower()

df['version'].fillna(df['version'].mode()[0],inplace=True)
df.drop('registered city',axis=1,inplace=True)

df['price'] = pd.to_numeric(df['price'], errors='coerce')
df.dropna(subset=['price'], inplace=True)

label_cols = ["make", "model", "version", "assembly", "transmission"]

le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])
import matplotlib.pyplot as plt

X = df.drop("price", axis=1)  
y = df["price"]                


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("-----------------------------")
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# print("\nSample Predictions:")
# print(y_pred[:10])
