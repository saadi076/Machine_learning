import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv(r"C:\Users\DELL\Desktop\ML dataset\KNN\cardekho_data.csv")
print(data.head(10))

categorical_columns = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

X = data.drop(columns=['Selling_Price'])
y = data['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred_class = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_class)
cm = confusion_matrix(y_test, y_pred_class)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(cm)

plt.figure(figsize=(20,10))
plot_tree(classifier, feature_names=X.columns, filled=True, class_names=True)
plt.show()
