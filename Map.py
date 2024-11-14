### 10. Apply Data analysis and representation on a Map using various Map datasets with user interaction.


import folium

# Create a map centered around a location
m = folium.Map(location=[20, 0], zoom_start=2)

# Adding markers with popup info
locations = [(51.505, -0.09, 'London'), (48.8566, 2.3522, 'Paris'), (40.7128, -74.0060, 'New York')]
for lat, lon, city in locations:
    folium.Marker([lat, lon], popup=city).add_to(m)

# Save the map
m.save('interactive_map.html')


### 16. Utilize Data analysis and representation on a Map using various Map datasets with Mouse Rollover effect, user interaction, etc.


import folium

# Create a map centered around a location
m = folium.Map(location=[20, 0], zoom_start=2)

# Adding markers with mouse rollover effect
locations = [(51.505, -0.09, 'London'), (48.8566, 2.3522, 'Paris'), (40.7128, -74.0060, 'New York')]
for lat, lon, city in locations:
    folium.Marker([lat, lon], popup=city, tooltip='Click for info').add_to(m)

# Save the map
m.save('interactive_map_with_tooltip.html')


### 29. Apply Data analysis and representation on a Map using various Map datasets with Mouse Rollover effect.


import folium

# Create a map centered around a location
m = folium.Map(location=[20, 0], zoom_start=2)

# Adding markers with mouse rollover effect
locations = [(51.505, -0.09, 'London'), (48.8566, 2.3522, 'Paris'), (40.7128, -74.0060, 'New York')]
for lat, lon, city in locations:
    folium.Marker([lat, lon], popup=city, tooltip='Click for info').add_to(m)

# Save the map
m.save('interactive_map_with_tooltip.html')


### 31. Build cartographic visualization for multiple datasets involving various countries of the world, states, and districts in India, etc.


import folium

# Create a world map
world_map = folium.Map(location=[20, 0], zoom_start=2)

# Adding markers for various countries
countries = [(20, 77, 'India'), (35.8617, 104.1954, 'China'), (51.1657, 10.4515, 'Germany')]
for lat, lon, country in countries:
    folium.Marker([lat, lon], popup=country).add_to(world_map)

# Save the world map
world_map.save('world_map_with_countries.html')
