import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#importing datset
dataset=pd.read_csv('filelocation/Admit_data.csv')
X=dataset.iloc[:,1:8].values 
Y=dataset.iloc[:,8].values

for i in range(0,400):
    if(Y[i]>=0.5):
        Y[i]=1
    else:
        Y[i]=0
    
dataset=dataset.drop(dataset.columns[[4,5,7]],axis=1)
#print(dataset)
#dataset.to_csv('C:/Users/Aju/Desktop/Admission-master/data.csv')

#Splitting the dataset into the training set and test set
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature Scaling 
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

# Applying PCA
pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

#Naive_Bayes
gnb=GaussianNB() 
gnb.fit(X_train,Y_train)
Y_pred=gnb.predict(X_test)
print ("\n\n ---Bayesian Classifier Model---")
gnb_roc_auc = accuracy_score(Y_test, gnb.predict(X_test))
print ("Bayesian Model AUC = %2.2f" % gnb_roc_auc)
print(classification_report(Y_test, gnb.predict(X_test)))

#Decision_Tree
dtree=DecisionTreeClassifier(criterion='entropy',random_state=0) 
dtree.fit(X_train,Y_train)
print ("\n\n ---Decision Tree Model---")
dt_roc_auc =accuracy_score(Y_test, dtree.predict(X_test))
print ("Decision Tree AUC = %2.2f" % dt_roc_auc)
print(classification_report(Y_test, dtree.predict(X_test)))
