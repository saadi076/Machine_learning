import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_csv(r"C:\Users\DELL\Desktop\Train.csv")
df.head(10)
df.drop('Unnamed: 0',inplace=True,axis=1)
df.head()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['TEXT']=le.fit_transform(df['TEXT'])
df.head(100)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
X=df.drop('Label',axis=1)
y=df['Label']
from sklearn.neighbors import KNeighborsRegressor
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential()
model.add(Dense(units=1024,activation='relu',input_dim=X_train.shape[1]))
model.add(Dense(units=512,activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1,activation='linear'))
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10,batch_size=32)
loss,accuracy=model.evaluate(X_test,y_test)
print(f"Loss:{loss},Accuracy:{accuracy}")
from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred)}")
model.summary()