import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\DELL\Desktop\ML dataset\drug200.csv")
print(data.head())
for X in data.columns[:-1]:
    plt.title(f'{X} vs Drug')
    plt.scatter(data[X],data['Drug'])
    plt.show()
   

label_encoder_sex = LabelEncoder()
data['Sex_encoded'] = label_encoder_sex.fit_transform(data['Sex'])
data['Bp_encoded'] = label_encoder_sex.fit_transform(data['Bp'])
print(data['Bp_encoded'].head(10))
X = data[['Age', 'Sex_encoded', 'Bp_encoded', 'Na_to_K']]
y = data['Drug']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
print(X_test.head(10))




