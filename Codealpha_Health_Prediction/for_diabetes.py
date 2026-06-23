import pandas as pd

data = pd.read_csv("diabetes.csv")
diabeticInput = data.drop("Outcome",axis = 1)
diabeticOutput = data["Outcome"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(diabeticInput,diabeticOutput,test_size = 0.2,random_state = 42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.svm import SVC
model_for_diabeties = SVC(kernel = "rbf",class_weight = 'balanced',random_state=42,probability=True)
model_for_diabeties.fit(x_train,y_train)


import joblib
joblib.dump(model_for_diabeties,"diabetes_model.pkl")
joblib.dump(scaler,"scaler_for_diabeties.pkl")


predictions = model_for_diabeties.predict(x_test)


'''
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
accuracy = accuracy_score(y_test,predictions)
precision = precision_score(y_test,predictions)
recall = recall_score(y_test,predictions)
f1 = f1_score(y_test,predictions)
print("Accuracy for Diabetes model :",accuracy*100,"%")
print("Precision for Diabetes model :",precision*100,"%")
print("Recall for Diabetes model :",recall*100,"%")
print("F1_Score for Diabetes model :",f1*100,"%")
print("Training accuracy :",model_for_diabeties.score(x_train,y_train)*100,"%")
print("Testing accuracy :",model_for_diabeties.score(x_test,y_test)*100,"%")
'''

