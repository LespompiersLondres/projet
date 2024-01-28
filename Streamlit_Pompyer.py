import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

Incident=pd.read_csv(r"C:\Users\agrabia\Documents\Formation\Projet/LFB Incident data Last 3 years.csv")
Mobilisation=pd.read_csv(r"C:\Users\agrabia\Documents\Formation\Projet/LFB Mobilisation data Last 3 years.csv", sep=';')
Coordonnees_brigade=pd.read_csv(r"C:\Users\agrabia\Documents\Formation\Projet/LondonFireBrigade_Coordonnées.csv",sep=';')
Calendrier=pd.read_csv(r"C:\Users\agrabia\Documents\Formation\Projet/London_Pompyer_Calendrier.csv", sep=';')
Meteo=pd.read_excel(r"\Users\agrabia\Documents\Formation\Projet/london_weather2.xlsx")

st.title("Projet Pompyer")
st.sidebar.title("Sommaire")
pages=["Description du projet","Jeux de données","Visualisation","Préparation des données","Modélisation","Machine Learning","Conclusion"]
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
    
    st.subheader("La table 'Calendrier'")
    st.markdown("Cette table contient 1 247 enregistrements répartis sur 4 colonnes. Elle indique, pour chaque date entre le 1er janvier 2020 et le 31 mai 2023, s'il y a une caractéristique particulière. Ses colonnes sont :")
    st.markdown(""" 
                - DateOfCall_bis : (datetime) clé primaire
                - jour ferie : (int) variable binaire qui indique si le jour est férié ou non
                - vacances : (int) variable binaire qui indique si le jour est inclus dans des vacances scolaires ou non
                - COVID : (int) variable binaire qui indique si le jour faisait partie d'une période de confinement
                """)

    st.subheader("La table 'Coordonnees_brigade'")
    st.markdown("Cette table contient 109 enregistrements répartis sur 4 colonnes. Elle recense pour chaque caserne ses coordonnées géographiques. Ses colonnes sont :")
    st.markdown(""" 
                - DeployedFromStation_Name : (object) clé primaire
                - Adresse : (object) adresse complète de la caserne
                - Latitude : (float) latitude de la caserne
                - Longitude : (float) longitude de la caserne
                """)
    
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