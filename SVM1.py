import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r"C:\Users\DELL\Desktop\ML dataset\LogisticRegression\diabetes2.csv")

print(data.head(10))

feature1 = 'Glucose'
feature2 = 'BMI'

X = data[[feature1, feature2]]  
y = data['Outcome'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an SVM model
model = SVC(kernel='linear')

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = model.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy}")

# Plot decision boundary
plt.figure(figsize=(10, 6))

plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title('SVM Decision Boundary')

# Plot the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200), np.linspace(ylim[0], ylim[1], 200))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contour(xx, yy, Z, colors='k', levels=[0], linestyles=['-'])

plt.show()
