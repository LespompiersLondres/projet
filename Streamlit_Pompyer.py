import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn import linear_model,preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

Incident=pd.read_csv(r"C:\Users\agrabia\Documents\Formation\Projet/LFB Incident data Last 3 years.csv")
Mobilisation=pd.read_csv(r"C:\Users\agrabia\Documents\Formation\Projet/LFB Mobilisation data Last 3 years.csv", sep=';')
Coordonnees_brigade=pd.read_csv(r"C:\Users\agrabia\Documents\Formation\Projet/LondonFireBrigade_Coordonnées.csv",sep=';')
Calendrier=pd.read_csv(r"C:\Users\agrabia\Documents\Formation\Projet/London_Pompyer_Calendrier.csv", sep=';')
Meteo=pd.read_excel(r"\Users\agrabia\Documents\Formation\Projet/london_weather2.xlsx")
Final_1=pd.read_csv(r"C:\Users\agrabia\Documents\Formation\Projet/Final_1_corrt.csv")

st.title("Projet Pompyer")
st.sidebar.title("Sommaire")
pages=["Description du projet","Jeux de données","Visualisation","Préparation des données et Modélisation : Modèle à variables minimales","Préparation des données et Modélisation : Modèle à variables maximales" ,"Machine Learning : Modèle à variables minimales","Machine Learning : Modèle à variables maximales","Conclusion","Cartographie"]
page=st.sidebar.radio("Aller vers",pages)

if page==pages[0]:
    st.image('London_Fire_Brigade_FRU42_-_WX71_YJV.jpg')
    st.write('Crédit image : Hullian111, CC-BY-SA 4.0, [via Wikimedia Commons](https://commons.wikimedia.org/wiki/File:London_Fire_Brigade_FRU42_-_WX71_YJV.jpg)')
    st.header("Description du projet")
    st.markdown(
        "Ce projet a été réalisé dans le cadre de la formation Data Scientist de l'organisme de formation Data Scientest. L’objectif de ce projet est d’estimer les temps de trajet de la Brigade des Pompiers de Londres. Cette brigade est le service d'incendie et de sauvetage le plus actif du Royaume-Uni  et l'une des plus grandes organisations de lutte contre l'incendie et de sauvetage au monde.") 
    st.markdown("Cette application présente notre démarche, de l'exploration des jeux de données jusqu'au Machine Learning.")

if page==pages[1]:
    st.header("Jeux de données")
    st.subheader('Données à disposition')
    st.markdown("Nous avions à disposition deux jeux de données issus du site data.london.gov.uk.")
    st.markdown("Le **premier jeu de données** est une base des incidents qui décrit les détails de chaque incident auquel ont assisté les pompiers de Londres sur les années pleines 2020-2022 et janvier-mai 2023. Les informations fournies permettent de savoir quand et où l'incident s'est produit ainsi que son type.(https://data.london.gov.uk/dataset/london-fire-brigade-incident-records)")
    st.markdown("Le **second jeu de données** est une base de mobilisation qui contient les détails de chaque camion de pompiers (appareil de pompage) envoyé sur l'incident au cours de la même période. Les informations sont fournies pour l'appareil mobilisé, d'où il a été déployé et les heures enregistrées pour arriver à l'incident.(https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records)")
    st.subheader('Données externes')
    st.markdown("Il nous a semblé pertinent de récupérer des **données météorologiques** en partant du postulat que de mauvaises conditions météorologiques auraient un impact sur le temps de trajet.(https://www.visualcrossing.com/weather-history/London,UK)")
    st.markdown("La seconde donnée pertinente à récupérer concerne le **calendrier scolaire**. Les périodes de congés sont généralement des moments avec moins de traffic (en dehors des départs et retours de vacances). De même, les dates des **jours fériés** ont été récupérées.")
    st.markdown("2020 a été une année de pandémie avec des confinements décrétés. Nous avons donc pris en compte les **dates de confinement** dans notre modèle.")
    st.markdown("Enfin, la distance à parcourir entre la caserne et le lieu de l'incident fait également partie des données à prendre en compte. Nous avons donc récupéré les **coordonnées géographiques de chaque caserne** via Google Maps.")
    st.image('JDD.jpg')

    st.subheader("La table 'Incident'")
    st.markdown("Cette table contient 381 366 enregistrements répartis sur 39 colonnes. Chaque incident est unique au sein de cette table. Ses colonnes sont :")
    st.markdown("""
                - IncidentNumber : (object)	clé primaire
                - DateOfCall : (object)	date de l'appel
                - CalYear : (int) année de l'appel
                - TimeOfCall : (object)	heure de l'appel avec découpage heure, minutes, secondees
                - HourOfCall : (int) heure de l'appel (incident)
                - IncidentGroup : (object)	type d'incident
                - StopCodeDescription : (object) détail du type d'incident
                - SpecialServiceType : (object)	détail de la modalité SpecialService
                - PropertyCategory : (object) catégorie du lieu de l'incident
                - PropertyType : (object) découpage plus fin de la catégorie du lieu de l'incident
                - AddressQualifier : (object) correspondance entre l'adresse indiquée et l'adresse réelle de l'incident
                - Postcode_full	: (object) code postal du lieu de l'incident
                - Postcode_district	: (object) code postal du district du lieu de l'incident
                - UPRN : (int) Unique Property Reference Number
                - USRN : (int) Unique Street Reference Number
                - IncGeo_BoroughCode : (object)	Code de l'arrondissement de l'incident
                - IncGeo_BoroughName: (object) Nom de l'arrondissement de l'incident en majuscules
                - ProperCase : (object)	Nom de l'arrondissement de l'incident en minuscules
                - IncGeo_WardCode : (object) Code du quartier de l'incident
                - IncGeo_WardName : (object) Nom du quartier de l'incident
                - IncGeo_WardNameNew : (object)	Nom du quartier de l'incident
                - Easting_m : (float) Latitude
                - Northing_m : (float) Longitude
                - Easting_rounded : (float)	Latitude
                - Northing_rounded : (float) Longitude
                - Latitude : (float) Latitude
                - Longitude : (float) Longitude
                - FRS : (object) Ville dont dépend la brigade, ici London
                - IncidentStationGround	: (object) Caserne rattachée au lieu de l'incident
                - FirstPumpArriving_AttendanceTime : (float) temps de réponse du premier véhicule
                - FirstPumpArriving_DeployedFromStation : (object) caserne de provenance du premier véhicule
                - SecondPumpArriving_AttendanceTime	: (float) temps de réponse du second véhicule
                - SecondPumpArriving_DeployedFromStation : (object)	caserne de provenance du second véhicule
                - NumStationsWithPumpsAttending : (float) nombre de casernes avec au moins un véhicule envoyé
                - NumPumpsAttending	: (float) nombre de véhicules envoyés
                - PumpCount : (float) nombre total de véhicules
                - PumpHoursRoundUp : (float) temps passé sur l'incident arrondi à l'heure
                - Notional Cost (£) : (float) coût de l'incident
                - Numcalls : (float) nombre d'appels pour un incident
                """)
    if st.checkbox("Afficher les 10 premières lignes de la table Incident"):
        st.dataframe(Incident.head(10))
 
    st.subheader("La table 'Mobilisation'")
    st.markdown("Cette table contient 560 530 enregistrements répartis sur 22 colonnes. Elle recence, pour chaque incident, le premier véhicule mobilisé et éventuellement le second. Ses colonnes sont :")
    st.markdown(""" 
                - IncidentNumber : (object) clé étrangère avec la table 'Incident'
                - CalYear : (int) année de l'appel
                - HourOfCall : (int) heure de l'appel
                - ResourceMobilisationId : (int) clé primaire
                - Resource_Code : (object) Code de la ressource
                - PerformanceReporting : (object) Identification du 1er et du 2ème véhicule mobilisé
                - DateAndTimeMobilised : (object) Date et heure de mobilisation
                - DateAndTimeMobile : (object) date et heure de départ de la caserne
                - DateAndTimeArrived : (object)	date et heure d'arrivée sur les lieux de l'incident
                - TurnoutTimeSeconds : (float) temps de mobilisation
                - TravelTimeSeconds	: (float) temps de trajet / variable cible
                - AttendanceTimeSeconds : (int)	temps de réponse (TurnoutTimeSeconds+TravelTimeSeconds)
                - DateAndTimeLeft : (object) Date et heure de départ de l'incident
                - DateAndTimeReturned : (float)	Date et heure de retour à la caserne
                - DeployedFromStation_Code : (object) Code de la caserne du véhicule
                - DeployedFromStation_Name : (object) clé étrangère avec la table 'Coordonnees_brigade'
                - DeployedFromLocation : (object) Véhicule envoyé de sa caserne ou d'une autre caserne
                - PumpOrder : (int)	Ordre d'arrivée des véhicules
                - PlusCode_Code : (object)	Pas d'information sur la variable
                - PlusCode_Description : (object) Pas d'information sur la variable	
                - DelayCodeId : (float)	Code du retard
                - DelayCode_Description : (object)	Description du retard
                """)
    if st.checkbox("Afficher les 10 premières lignes de la table Mobilisation"):
        st.dataframe(Mobilisation.head(10))
    
    st.subheader("La table 'Calendrier'")
    st.markdown("Cette table contient 1 247 enregistrements répartis sur 4 colonnes. Elle indique, pour chaque date entre le 1er janvier 2020 et le 31 mai 2023, s'il y a une caractéristique particulière. Ses colonnes sont :")
    st.markdown(""" 
                - DateOfCall_bis : (datetime) clé primaire
                - jour ferie : (int) variable binaire qui indique si le jour est férié ou non
                - vacances : (int) variable binaire qui indique si le jour est inclus dans des vacances scolaires ou non
                - COVID : (int) variable binaire qui indique si le jour faisait partie d'une période de confinement
                """)
    if st.checkbox("Afficher les 10 premières lignes de la table Calendrier"):
        st.dataframe(Calendrier.head(10))

    st.subheader("La table 'Coordonnees_brigade'")
    st.markdown("Cette table contient 109 enregistrements répartis sur 4 colonnes. Elle recense pour chaque caserne ses coordonnées géographiques. Ses colonnes sont :")
    st.markdown(""" 
                - DeployedFromStation_Name : (object) clé primaire
                - Adresse : (object) adresse complète de la caserne
                - Latitude : (float) latitude de la caserne
                - Longitude : (float) longitude de la caserne
                """)
    if st.checkbox("Afficher les 10 premières lignes de la table Coordonnees_brigade"):
        st.dataframe(Coordonnees_brigade.head(10))
    
    st.subheader("La table 'Meteo'")
    st.markdown("Cette table contient 1 441 enregistrements répartis sur 6 colonnes. Elle mentionne pour chaque date les principales caractéristiques météorologiques. Ses colonnes sont :")
    st.markdown(""" 
                - DateOfCall_bis : (datetime)) clé primaire
                - tempmax : (float) température maximale
                - tempmin : (float) température minimale
                - temp : (float) température moyenne
                - precip : (float) niveau de précipitation
                - snow : (float) cm de neige
                """)
    if st.checkbox("Afficher les 10 premières lignes de la table Meteo"):
        st.dataframe(Meteo.head(10))

if page==pages[2]:
    st.header("Visualisation")
    st.markdown("Voici quelques modélisations issues des .")
    st.subheader("Répartition des incidents par mois et par heures d’appel")
    Incident['DateOfCall_bis'] = pd.to_datetime(Incident['DateOfCall'])
    Incident["MonthOfCall"] = Incident['DateOfCall_bis'].dt.month
    Incident_gprmonth = Incident.groupby(["MonthOfCall","CalYear"]).count().reset_index() 
    data=Incident_gprmonth
    year_options=data["CalYear"].unique()
    selected_year=st.selectbox("Sélectionnez une année",options=year_options,index=0)
    filtered_data=data[data["CalYear"]==selected_year]
    fig=px.line(filtered_data, x="MonthOfCall", y="IncidentNumber",animation_frame="CalYear")
    st.plotly_chart(fig)
    #fig=plt.figure()
    #sn.lineplot(data=Incident_gprmonth, x="MonthOfCall", y="IncidentNumber", hue="CalYear", style="CalYear",dashes=False,markers=True,sizes=(.25, 2.5))
    #st.pyplot(fig)
    #fig2=plt.figure()
    #Incident['HourOfCall'].value_counts(normalize=True).plot(kind='bar')
    #st.pyplot(fig2)
    st.image('Répartition_Heures.jpg')
    st.subheader("Répartition des incidents par catégorie")
    #fig3=plt.figure()
    #Incident['IncidentGroup'].value_counts(normalize=True).plot(kind='pie')
    #st.pyplot(fig3)
    #st.image('Répartition_catégorie.jpg')
    labels=['False Alarm','Fire','Special Service']
    values=[186923,57697,136746]
    fig3=go.Figure(data=[go.Pie(labels=labels,values=values)])
    st.plotly_chart(fig3)
    Incident['StopCodeDescription'].value_counts(normalize=True)*100
    st.subheader("Répartition des incidents par caserne")
    st.image('Caserne.jpg')
    st.image('Caserne2.jpg')
    st.subheader("Boîte à moustache du temps de trajet par heure (données 2020)")
    st.image('Heures.jpg')

if page==pages[3]:
    st.header("Préparation des données et Modélisation : Modèle à variables minimales ")

if page==pages[4]:
    st.header("Préparation des données et Modélisation : Modèle à variables maximales")
    st.markdown("""
                - Une des premières remarques lors de l'observation des deux tables à disposition est la prépondérance des variables sous format Objet qui ne peuvent donc pas être traitées aisément en machine learning. 
                - Ensuite, la relation entre la table Incident et la table Mobilisation est de l'ordre de 1 pour N, plusieurs équipages ayant pu être mobilisés sur une même intervention. Pour une meilleure cohérence dans les temps d'arrivée, seuls les premiers équipages ont été conservés donc variable PerformanceReporting à 1.
                - Une fois la jointure faite entre la table Incident et la table Mobilisation, plusieurs suppression de variables vides, redondantes ou n'ayant a priori pas d'impact sur le temps d'arrivée ont été réalisées. """)
    
    if st.checkbox("Afficher toutes les suppressions"):
        st.markdown("""
                    - 'StopCode_Description' : Variable trop détaillée qui n'apporte pas grand-chose
                    - Suppresion de la variable 'DateAndTimeReturned' vide
                	- 'SpecialServiceType' : trop de valeurs nulles
                	- 'PropertyCategory' : pas d'utilité pour le projet
                	- 'PropertyType' : trop de catégories différentes
                	- 'AddressQualifier' : variable qui ne peut être renseignée qu'a posteriori donc non utile pour le Machine Learning
                	- 'Postcode_full' : trop de NA
                	- 'Postcode_district' : trop fin
                	- 'UPRN' et 'USRN' : pour la première, trop de NA, pour la seconde, trop de modalités distinctes
                	- 'IncGeo_BoroughName' : on ne conserve que le code de l'arrondissement et non son nom
                	- 'ProperCase' : idem que la variable précédente
                	- 'IncGeo_WardCode','IncGeo_WardName','IncGeo_WardNameNew' : les variables liées au quartier n'ont pas été conservées
                	- 'Easting_m','Northing_m' : Trop de NA
                	- 'FRS' : une seule modalité donc inutile
                	- 'IncidentStationGround' : redondant avec 'DeployedFromStation_Name'
                	- 'FirstPumpArriving_AttendanceTime','SecondPumpArriving_AttendanceTime' : pas d'apport
                	- 'FirstPumpArriving_DeployedFromStation','SecondPumpArriving_DeployedFromStation' : redondant avec 'DeployedFromStation_Name'
                	- 'NumStationsWithPumpsAttending' : on se concentre sur la première brigade arrivée
                	- 'NumPumpsAttending' : on se concentre sur la première brigade arrivée
                	- 'PumpCount' : on se concentre sur la première brigade arrivée
                	- 'PumpHoursRoundUp' : Pas d'apport sur la variable cible de durée du trajet
                	- 'Notional Cost (£)' : pas d'info pertinente
                	- 'Numcalls' : pas d'apport sur le projet
                	- 'ResourceMobilisationId' : clé primaire de la table Mobilisation qui n'a plus d'utilité dans la table fusionnée
                	- 'Resource_Code' : trop de modalités différentes
                	- 'PerformanceReporting' : filtre sur la modalité 1
                	- 'DateAndTimeMobilised','DateAndTimeMobile','DateAndTimeArrived', 'DateAndTimeLeft' : pas d'apport de l'heure exacte
                	- 'TurnoutTimeSeconds' : pas d'impact sur la variable cible
                	- 'AttendanceTimeSeconds' : regroupe temps de préparation + temps de trajet donc redondant avec la variable cible
                	- 'DeployedFromStation_Code' : c'est le nom et non le code qui a été conservé
                	- 'DeployedFromLocation' : pas d'utilité
                	- 'PumpOrder' : on se concentre sur la première brigade arrivée
                	- 'PlusCode_Code', 'PlusCode_Description' : une seule modalité donc inutile
                	- 'DelayCode_Description' : le code a été conservé et non le nom
                    - 'Latitude' et 'Longitude' : d'abord utilisées, elles comportaient trop de NA pour être vraiment exploitables
                    """)
    st.markdown("""
                - Plusieurs variables, notamment temporelles, ont également été créées. """)
    if st.checkbox("Afficher toutes les créations"):
        st.markdown("""
                    - Création d'une variable 'DateOfCall_bis', format date de la variable 'DateOfCall'
                    - Création d'une variable 'MinuteOfCall' à partir de la variable 'TimeOfCall'
                    - Création d'une variable 'Joursem' à partir de la variable 'DateOfCall_bis'
                    - Création d'une variable 'mois' à partir de la variable 'DateOfCall__bis'
                    - Création d'une variable 'jour' à partir de la variable 'DateOfCall_bis'
                    """)
    st.markdown("""
                - Les valeurs vides de la variable 'DelayCodeId' ont été remplacées par la valeur 12 qui équivaut à pas de retard 
                - Les modalités de la variable 'IncidentGroup^' ont été transformées en variable numérique
                - Une fois ces modifications effectuées, cette table principale a été jointe aux tables Calendrier et Coordonnees_Brigade
                - Le package BNG a permis de transformer les coordonnées géographiques britanniques en latitude et longitude. Le package Geopy a ensuite été utilisé pour calculer les distances entre le lieu de l'incident et les coordonnées de la caserne
                - Les quelques NA présents sur la variable cible 'TravelTimeSeconds' ont été supprimés
                - Le temps moyen de parcours en fonction de la caserne et de l'heure de l'appel ainsi que le temps moyen de parcours en fonction du quartier et de l'heure de départ ont été calculés. Les deux nouvelles variables ainsi créées ont pu remplacer les variables 'IncGeo_BoroughCode' et 'DeployedDromStation_Name'
                - Enfin, la table principale a été jointe avec la table Meteo et la table finale a été exportée (Final_1)
                """)
    if st.checkbox("Afficher les 10 premières lignes de la table Final_1"):
        st.dataframe(Final_1.head(10))
    
if page==pages[5]:
    st.header("Machine Learning : Modèle à variables minimales")

if page==pages[6]:
    st.header("Machine Learning : Modèle à variables maximales")
    st.subheader("Temps de trajet")
    Final_1=Final_1.dropna()
    data=Final_1.drop(['TravelTimeSeconds','IncidentNumber','DateOfCall_bis','DelayCodeId_bis'], axis=1)
    target=Final_1['TravelTimeSeconds']
    X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=66)
    X_train_scaled=preprocessing.scale(X_train)
    scaler=preprocessing.StandardScaler().fit(X_train)
    X_train_scaled=scaler.transform(X_train)
    scaler=preprocessing.StandardScaler().fit(X_test)
    X_test_scaled=scaler.transform(X_test)

    knn_tp=joblib.load("model_knn_tp")
    rf_tp=joblib.load("model_rf_tp")
    ac_tp=joblib.load("model_ac_tp")
    
    y_pred_knn_tp=knn_tp.predict(X_test_scaled)
    y_pred_rf_tp=rf_tp.predict(X_test_scaled)
    y_pred_ac_tp=ac_tp.predict(X_test_scaled)
    

    model_choisi=st.selectbox(label="Modèle",options=['KNeighbors Regressor','Random Forest Regressor','DecisionTree Regressor'])

    @st.cache_data
    def train_model(model_choisi):
        if model_choisi=='KNeighbors Regressor':
            y_pred=y_pred_knn_tp
            score_tp=knn_tp.score(X_test_scaled,y_test)
        elif model_choisi=='Random Forest Regressor':
            y_pred=y_pred_rf_tp
            score_tp=rf_tp.score(X_test_scaled,y_test)
        elif model_choisi=='DecisionTree Regressor':
            y_pred=y_pred_ac_tp
            score_tp=ac_tp.score(X_test_scaled,y_test)
        return score_tp

    st.write("score sans classe",train_model(model_choisi))

    st.subheader("5 classes")
    Final_1_5c=Final_1
    Final_1_5c['Class_duree']=pd.qcut(Final_1_5c['TravelTimeSeconds'],5,labels =[1,2,3,4,5])
    Final_1_5c=Final_1_5c.drop(['TravelTimeSeconds','IncidentNumber','DateOfCall_bis','DelayCodeId_bis'], axis=1)
    Final_1_5c=Final_1_5c.dropna()
    data_5c=Final_1_5c.drop(['Class_duree'], axis=1)
    target_5c=Final_1_5c['Class_duree']
    X_train_5c,X_test_5c,y_train_5c,y_test_5c=train_test_split(data_5c,target_5c,test_size=0.2,random_state=66)
    X_train_5c_scaled=preprocessing.scale(X_train_5c)
    scaler_5c=preprocessing.StandardScaler().fit(X_train_5c)
    X_train_5c_scaled=scaler_5c.transform(X_train_5c)
    scaler_5c=preprocessing.StandardScaler().fit(X_test_5c)
    X_test_5c_scaled=scaler_5c.transform(X_test_5c)

    knn_5c=joblib.load('model_knn_5c')
    rf_5c=joblib.load('model_rf_5c')
    ac_5c=joblib.load('model_ac_5c')
    
    y_pred_knn_5c=knn_5c.predict(X_test_5c_scaled)
    y_pred_rf_5c=rf_5c.predict(X_test_5c_scaled)
    y_pred_ac_5c=ac_5c.predict(X_test_5c_scaled)
    
    model_choisi_5c=st.selectbox(label="Modèle",options=['KNeighbors Classifier 5 classes','Random Forest Classifier 5 classes','DecisionTree Classifier 5 classes'])

    @st.cache_data
    def train_model_5c(model_choisi_5c):
        if model_choisi_5c=='KNeighbors Classifier 5 classes':
            y_pred_5c=y_pred_knn_5c
            score_5c=knn_5c.score(X_test_5c_scaled,y_test_5c)
        elif model_choisi_5c=='Random Forest Classifier 5 classes':
            y_pred_5c=y_pred_rf_5c
            score_5c=rf_5c.score(X_test_5c_scaled,y_test_5c)
        elif model_choisi_5c=='DecisionTree Classifier 5 classes':
            y_pred_5c=y_pred_ac_5c
            score_5c=ac_5c.score(X_test_5c_scaled,y_test_5c)
        return score_5c
    
    st.write("score 5 classes",train_model_5c(model_choisi_5c))

    @st.cache_data
    def crosstab_5c(model_choisi_5c):
        if model_choisi_5c=='KNeighbors Classifier 5 classes':
            crosstab_5c=pd.crosstab(y_test_5c,y_pred_knn_5c,rownames=['Classe réelle'], colnames=['Classe prédite'])
        elif model_choisi_5c=='Random Forest Classifier 5 classes':
            crosstab_5c=pd.crosstab(y_test_5c,y_pred_rf_5c,rownames=['Classe réelle'], colnames=['Classe prédite'])
        elif model_choisi_5c=='DecisionTree Classifier 5 classes':
            crosstab_5c=pd.crosstab(y_test_5c,y_pred_ac_5c,rownames=['Classe réelle'], colnames=['Classe prédite'])
        return crosstab_5c
    
    st.write("crosstab 5 classes",crosstab_5c(model_choisi_5c))


    st.subheader("3 classes")
    Final_1_3c=Final_1
    Final_1_3c['Class_duree']=pd.qcut(Final_1_3c['TravelTimeSeconds'],3,labels =[1,2,3])
    Final_1_3c=Final_1_3c.drop(['TravelTimeSeconds','IncidentNumber','DateOfCall_bis','DelayCodeId_bis'], axis=1)
    Final_1_3c=Final_1_3c.dropna()
    data_3c=Final_1_3c.drop(['Class_duree'], axis=1)
    target_3c=Final_1_3c['Class_duree']
    X_train_3c,X_test_3c,y_train_3c,y_test_3c=train_test_split(data_3c,target_3c,test_size=0.2,random_state=66)
    X_train_3c_scaled=preprocessing.scale(X_train_3c)
    scaler_3c=preprocessing.StandardScaler().fit(X_train_3c)
    X_train_3c_scaled=scaler_3c.transform(X_train_3c)
    scaler_3c=preprocessing.StandardScaler().fit(X_test_3c)
    X_test_3c_scaled=scaler_3c.transform(X_test_3c)

    knn_3c=joblib.load('model_knn_3c')
    rf_3c=joblib.load('model_rf_3c')
    ac_3c=joblib.load('model_ac_3c')

    y_pred_knn_3c=knn_3c.predict(X_test_3c_scaled)
    y_pred_rf_3c=rf_3c.predict(X_test_3c_scaled)
    y_pred_ac_3c=ac_3c.predict(X_test_3c_scaled)

    model_choisi_3c=st.selectbox(label="Modèle",options=['KNeighbors Classifier 3 classes','Random Forest Classifier 3 classes','DecisionTree Classifier 3 classes'])

    @st.cache_data
    def train_model_3c(model_choisi_3c):
        if model_choisi_3c=='KNeighbors Classifier 3 classes':
            y_pred_3c=y_pred_knn_3c
            score_3c=knn_3c.score(X_test_3c_scaled,y_test_3c)
        elif model_choisi_3c=='Random Forest Classifier 3 classes':
            y_pred_3c=y_pred_rf_3c
            score_3c=rf_3c.score(X_test_3c_scaled,y_test_3c)
        elif model_choisi_3c=='DecisionTree Classifier 3 classes':
            y_pred_3c=y_pred_ac_3c
            score_3c=ac_3c.score(X_test_3c_scaled,y_test_3c)
        return score_3c
    
    st.write("score 3 classes",train_model_3c(model_choisi_3c))

    @st.cache_data
    def crosstab_3c(model_choisi_3c):
        if model_choisi_3c=='KNeighbors Classifier 3 classes':
            crosstab_3c=pd.crosstab(y_test_3c,y_pred_knn_3c,rownames=['Classe réelle'], colnames=['Classe prédite'])
        elif model_choisi_3c=='Random Forest Classifier 3 classes':
            crosstab_3c=pd.crosstab(y_test_3c,y_pred_rf_3c,rownames=['Classe réelle'], colnames=['Classe prédite'])
        elif model_choisi_3c=='DecisionTree Classifier 3 classes':
            crosstab_3c=pd.crosstab(y_test_3c,y_pred_ac_3c,rownames=['Classe réelle'], colnames=['Classe prédite'])
        return crosstab_3c
    
    st.write("crosstab 3 classes",crosstab_3c(model_choisi_3c))
