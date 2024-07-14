import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\DELL\Desktop\ML dataset\LogisticRegression\Social_Network_Ads.csv")

print(data.head(10))

le = LabelEncoder()
data['Gender_Encoded'] = le.fit_transform(data['Gender'])

X = data[['Gender_Encoded', 'Age', 'EstimatedSalary']]
y = data['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

plt.figure(figsize=(10, 6))
plt.scatter(X_test['Age'], X_test['EstimatedSalary'], c=y_test, cmap='coolwarm', edgecolors='k')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Logistic Regression Decision Boundary')
plt.show()
