#meteo midas_uk


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import svm
from sklearn import ensemble
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor

merged_df=pd.read_csv("/home/user/Bureau/dataScientest/projet_pompiers_Londres/data/merged_meteo.csv")



target=merged_df['AttendanceTimeSeconds'] #à essayer avec Traveltime
df_train=merged_df[['cloud_cover', 'sunshine', 'mean_temp', 'precipitation', 'CalYear', 'HourOfCall']]

X_train, X_test, y_train, y_test = train_test_split(df_train, target, test_size=0.2)


# Concaténer X_train et y_train pour manipuler les données plus facilement
train_data = pd.concat([X_train, y_train], axis=1)

# Diviser la variable cible en quartiles
num_quartiles = 4  # Diviser en 4 quartiles

# Utilisation de la fonction qcut de Pandas pour diviser la cible en quartiles
train_data['target_quartile'] = pd.qcut(train_data[target.name], num_quartiles, labels=False)

# Après avoir ajouté une nouvelle colonne 'target_quartile' dans train_data, vous pouvez y accéder comme suit :
# train_data['target_quartile'] contient les quartiles correspondants à chaque ligne de votre ensemble d'entraînement.

# Maintenant, si vous voulez récupérer X_train et y_train avec la variable cible divisée en quartiles, vous pouvez le faire comme suit :
X_train = train_data.drop([target.name, 'target_quartile'], axis=1)
y_train = train_data['target_quartile']

# Votre modèle peut être formé sur X_train et y_train avec la variable cible divisée en quartiles.

############################################
score_knn=[]

for k in range(2, 41):
	knn=neighbors.KNeighborsRegressor(n_neighbors=k)
	knn.fit(X_train, y_train)
	score_knn.append(knn.score(X_test, y_test))
	

y_knn = knn.predict(X_test)


knn_score = knn.score(X_train, y_train)
print("KNeighborsRegressor score:", knn_score) 
#print(accuracy_score(y_test, y_knn))
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

#clf = ensemble.RandomForestClassifier(n_jobs=-1, random_state=321)
clf = RandomForestRegressor(max_depth=2, random_state=0)
clf.fit(X_train, y_train)

y_randomForest = clf.predict(X_test)

randomForest_score = clf.score(X_train, y_train)
print("RandomForestRegressor score:", randomForest_score) 
###############################################
#Com

#TBDdidviser la variable cible en classes ? (quartiles ?)
