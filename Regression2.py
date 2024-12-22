import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\DELL\Desktop\fake_bills.csv")
print(df.head(10))

df.fillna(df.mean(), inplace=True)
print(df.isnull().sum())
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X = df.drop('diagonal', axis=1)
y = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

sv = LinearRegression()

sv.fit(X_train, y_train)
y_pred = sv.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

def predict_diagonal(input_features):
    input_scaled = sc.transform([input_features])
    predicted_value = sv.predict(input_scaled)
    return predicted_value

new_data = [1, 104.48, 103.50, 4.40, 2.94, 113.16]  
predicted_diagonal = predict_diagonal(new_data)
print(f"Predicted Diagonal: {predicted_diagonal}")
actual = [172.69]
a1 = r2_score(actual, predicted_diagonal) 
print(f"Accuracy of predicted is {a1}")
from joblib import dump, load

dump(model, 'model.joblib')
loaded_model = load('model.joblib')
