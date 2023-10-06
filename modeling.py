#meteo midas_uk


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import svm
from sklearn import ensemble

merged_df=pd.read_csv("/home/lva/Bureau/dataScientest/projet_pompiers_Londres/data/merged_inner.csv")



target=merged_df['AttendanceTimeSeconds'] #à essayer avec Traveltime
df_train=merged_df[['HourOfCall', 'CalYear']]

X_train, X_test, y_train, y_test = train_test_split(df_train, target, test_size=0.2)

############################################
score_knn=[]

for k in range(2, 41):
	knn=neighbors.KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train, y_train)
	score_knn.append(knn.score(X_test, y_test))
	

y_knn = knn.predict(X_test)
	
#############################################

clf = svm.SVC(gamma=0.01,  kernel='poly')
clf.fit(X_train, y_train)
parametres = {'C':[0.1,1,10,30], 'kernel':['linear', 'sigmoid', 'rbf'], 'gamma':[0.001, 0.1, 0.5]}
grid_clf = model_selection.GridSearchCV(estimator=clf, param_grid=parametres)
grille = grid_clf.fit(X_train,y_train)
print(pd.DataFrame.from_dict(grille.cv_results_).loc[:,['params', 'mean_test_score']]) 
#Choisir gamma et kernel
clf = svm.SVC(gamma=0.01,  kernel='linear')
clf.fit(X_train, y_train)
y_svc=clf.predict(X_test)

############################################
#TBD
#    'max_features': "sqrt", "log2", None
#    'min_samples_split': Nombres pairs allant de 2 à 30

clf = ensemble.RandomForestClassifier(n_jobs=-1, random_state=321)
clf.fit(X_train, y_train)

y_randomForest = clf.predict(X_test)


###############################################
#Comparaison
from sklearn.metrics import confusion_matrix


pd.crosstab(y_test, y_knn, rownames=['Classe réelle'], colnames=['Classe prédite knn'])
pd.crosstab(y_test, y_svc, rownames=['Classe réelle'], colnames=['Classe prédite svc'])
pd.crosstab(y_test, y_randomForest, rownames=['Classe réelle'], colnames=['Classe prédite random forest'])

