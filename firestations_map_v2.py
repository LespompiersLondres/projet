import folium
import pandas as pd
df_fs=pd.read_csv("/home/user/Bureau/dataScientest/projet_pompiers_Londres/data/firestations.csv")

merged_df=pd.read_csv("/home/user/Bureau/dataScientest/projet_pompiers_Londres/data/merged_inner.csv")

merged_df_for_map = pd.merge(df_fs, df_from_merged_target, left_on='firestation_name', right_on='IncidentStationGround', how='inner')

merged_df_for_map.drop('IncidentStationGround', axis=1, inplace=True)



london_center = [51.5074, -0.1278]
zoom_level = 11

# Créer la carte centrée sur Londres
mymap = folium.Map(location=london_center, zoom_start=zoom_level)

# Parcourir les points et ajouter des marqueurs à la carte
for i in range(len(merged_df_for_map)):
    popup_content = f"<b>{merged_df_for_map['firestation_name'][i]}</b><br><i>temps moyen en mn:</i><b>{round(merged_df_for_map['meanTravelTimeMinutes'][i], 2)}</b>"
    folium.Marker(location=[merged_df_for_map['firestation_latitude'][i], merged_df_for_map['firestation_longitude'][i]], popup=popup_content).add_to(mymap)

# Afficher la carte
mymap.save('/home/user/Bureau/dataScientest/projet_pompiers_Londres/code/carte_interactive/map.html')



