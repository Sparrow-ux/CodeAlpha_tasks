import pandas as pd

data = pd.read_csv("dataset.csv")

inputs = data.drop("target",axis=1)
output = data["target"]

import sklearn
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(inputs,output,test_size=0.2,random_state = 42)

import xgboost
from xgboost import XGBClassifier

model_for_heart = XGBClassifier(n_estimators = 200, max_depth = 3, learning_rate = 0.01, random_state = 42)
model_for_heart.fit(x_train,y_train)


import joblib
joblib.dump(model_for_heart,"heart_disease_model.pkl")


predictions = model_for_heart.predict(x_test)
result = model_for_heart.predict_proba(x_test)

'''
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
accuracy = accuracy_score(y_test,predictions)
precision = precision_score(y_test,predictions)
recall = recall_score(y_test,predictions)
f1 = f1_score(y_test,predictions)
print("Accuracy for Heart disease model :",accuracy*100,"%")
print("Precision for Heart disease model :",precision*100,"%")
print("Recall for Heart disease model :",recall*100,"%")
print("F1_Score for Heart disease model :",f1*100,"%")
important = pd.DataFrame({"Feature":inputs.columns,"Importance":model_for_heart.feature_importances_})
important = important.sort_values(by = "Importance",ascending = False)
print(important)
print("Training accuracy :",model_for_heart.score(x_train,y_train)*100,"%")
print("Testing accuracy :",model_for_heart.score(x_test,y_test)*100,"%")
'''
