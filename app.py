import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv(r"C:\Users\DELL\Desktop\punjab-crime-stats.csv")
df = data
print(df.head(10))

label_encoder = LabelEncoder()
df['Type'] = label_encoder.fit_transform(df['CrimeType'])
df['City'] = label_encoder.fit_transform(df['District'])
df['City1'] = label_encoder.fit_transform(df['Division'])

print(df.head())

features = df[['Type', 'City1', 'City']]
target = df['CrimeCount']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

plt.scatter(y_test, y_pred)
plt.xlabel('Actual CrimeCount')
plt.ylabel('Predicted CrimeCount')
plt.title('Actual vs Predicted CrimeCount using Decision Tree Regressor')
plt.show()
