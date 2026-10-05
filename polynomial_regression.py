import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('lr', LinearRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\nPolynomial Regression (Train-Test Split):")
print("----------------------------------")
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='r2')
mae_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_mean_absolute_error')
mse_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_mean_squared_error')

print("\nPolynomial Regression with K-Fold (k=5):")
print("----------------------------------")
print("Average R² Score:", r2_scores.mean())
print("Average MAE:", -mae_scores.mean())
print("Average MSE:", -mse_scores.mean())
print("Average RMSE:", np.sqrt(-mse_scores.mean()))
