import numpy as np
import pandas as pd
import pickle
# from sklearn.model_selection import train_test_split,cross_val_score
# from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('always') 

data = pd.read_csv("phishing.csv")
X = data.drop(columns=["Index","class"])
X = X.drop(columns=["DomainRegLen","AgeofDomain","DNSRecording"])

Y=data["class"]
Y=pd.DataFrame(Y)
print(Y)
#Decision Tree Classifier
tree = DecisionTreeClassifier()
tree = tree.fit(X,Y)

# tree_predict = tree.predict(test_X)
pickle.dump(tree, open('model.pkl','wb'))


# model = pickle.load(open('model.pkl','rb'))
# pre=model.predict([[12,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])
# print(pre)