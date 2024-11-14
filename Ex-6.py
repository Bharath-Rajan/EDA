21) Perform Data analysis and representation on a Map using various Map data sets with Mouse Rollover effect, user interaction , etc.,

import folium
import pandas as pd
from folium.plugins import HeatMap

# Sample data for plotting (latitude, longitude, and popup information)
data = pd.DataFrame({
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Latitude': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484],
    'Longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740],
    'Population': [8419000, 3980400, 2706000, 2328000, 1690000]
})

# Create a base map centered on the US with OpenStreetMap tiles (default)
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles="OpenStreetMap")

# Add a marker with popup and mouse rollover effect for each city
for i, row in data.iterrows():
    popup_info = f"<strong>City:</strong> {row['City']}<br><strong>Population:</strong> {row['Population']}"
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup_info,
        tooltip=row['City'],  # Tooltip displays on mouse rollover
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

# Add a Heatmap overlay to show population concentration
heat_data = [[row['Latitude'], row['Longitude'], row['Population']] for i, row in data.iterrows()]
HeatMap(heat_data, radius=15).add_to(m)

# Option 1: Display the map inline in Jupyter Notebook
m

# Option 2: Save the map as an HTML file and open it in a browser
m.save("interactive_map.html")
print("Map saved as 'interactive_map.html'")
============================================================================================================================================================================================
