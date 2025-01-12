import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv(r"C:\Users\DELL\Desktop\data.csv")
data.drop('Unnamed: 32', axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Train and evaluate each classifier
accuracy_scores = {}
conf_matrices = {}
predictions = {}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = accuracy
    conf_matrices[name] = confusion_matrix(y_test, y_pred)
    predictions[name] = clf.predict(X_test)

    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

# Input prediction for all classifiers
input_text = [1009, 12.36, 21.8, 79.78, 466.1, 0.08772, 0.09445, 0.06015, 0.03745, 0.193, 0.06404, 
              0.2978, 1.502, 2.203, 20.95, 0.007112, 0.02493, 0.02703, 0.01293, 0.01958, 0.004463, 
              13.83, 30.5, 91.46, 574.7, 0.1304, 0.2463, 0.2434, 0.1205, 0.2972, 0.09261]
np_df = np.asarray(input_text).reshape(1, -1)
np_df = sc.transform(np_df)

print("\nPredictions for input data:")
for name, clf in classifiers.items():
    prediction = clf.predict(np_df)
    print(f"{name}: {'Cancerous' if prediction[0] == 1 else 'Not cancerous'}")

# Plot accuracies
plt.figure(figsize=(10, 6))
plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color='skyblue')
plt.title('Accuracy of Different Classifiers')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()

# Plot heatmaps for all confusion matrices
for name, conf_matrix in conf_matrices.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Pairplot for accuracy comparison (optional visualization)
accuracy_df = pd.DataFrame({'Classifier': accuracy_scores.keys(), 'Accuracy': accuracy_scores.values()})
sns.barplot(data=accuracy_df, x='Accuracy', y='Classifier', palette='viridis')
plt.title('Classifier Accuracy Comparison')
plt.show()
