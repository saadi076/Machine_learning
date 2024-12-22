import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"C:\Users\DELL\Desktop\data.csv")
data.drop('Unnamed: 32', axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

sv = SVC()

sv.fit(X_train, y_train)
y_pred = sv.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

input_text = [1009,12.36, 21.8, 79.78, 466.1, 0.08772, 0.09445, 0.06015, 0.03745, 0.193, 0.06404, 0.2978, 1.502, 2.203, 20.95, 0.007112, 0.02493, 0.02703, 0.01293, 0.01958, 0.004463, 13.83, 30.5, 91.46, 574.7, 0.1304, 0.2463, 0.2434, 0.1205, 0.2972, 0.09261
]

np_df = np.asarray(input_text).reshape(1, -1)
np_df = sc.transform(np_df)
prediction = sv.predict(np_df)


print("Cancerous" if prediction[0] == 1 else "Not cancerous!")
pred_accuracy = accuracy_score([1], prediction) 
print(f"Accuracy of predicted input text values: {pred_accuracy:.4f}")

