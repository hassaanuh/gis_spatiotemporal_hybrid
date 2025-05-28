import pandas as pd
import numpy as np
import folium
from folium import plugins
import warnings
warnings.filterwarnings('ignore')

print('=== Creating KSC Model Comparison Map ===')

# Load the prediction results
predictions_df = pd.read_csv('ksc_model_predictions.csv')
print(f'Loaded {len(predictions_df)} prediction records')

# Load the original data to get coordinates
files = {
    'PM2.5': 'monthly/merged_2.5_monthly.csv',
    'CO': 'monthly/merged_co_monthly.csv',
    'SO2': 'monthly/merged_so2_monthly.csv',
    'NO': 'monthly/merged_no_monthly.csv',
    'PM10': 'monthly/merged_pm10_monthly.csv'
}

# Load and combine data to get site coordinates
dfs = []
for name, file in files.items():
    try:
        df = pd.read_csv(file)
        df['pollutant'] = name
        dfs.append(df)
    except Exception as e:
        print(f'Error loading {file}: {e}')

combined_df = pd.concat(dfs)
pivot_df = combined_df.pivot_table(
    index=['Local Site Name', 'Month_Year', 'Site Latitude', 'Site Longitude'],
    columns='pollutant', 
    values='monthly_value'
)

df_reset = pivot_df.reset_index()
df_clean = df_reset.dropna(subset=['PM2.5', 'Site Latitude', 'Site Longitude'])
df_clean['Month_Year'] = pd.to_datetime(df_clean['Month_Year'])
df_clean = df_clean.sort_values(['Local Site Name', 'Month_Year'])

# Get unique sites with their coordinates
site_coords = df_clean.groupby('Local Site Name').agg({
    'Site Latitude': 'first',
    'Site Longitude': 'first'
}).reset_index()

print(f'Found {len(site_coords)} unique monitoring sites')

# Create sequences to match the test data
def create_sequences_with_coords(site_data, seq_length=6):
    sequences = []
    targets = []
    coords = []
    
    site_data = site_data.sort_values('Month_Year')
    
    pollutant_cols = ['PM2.5', 'CO', 'SO2', 'NO', 'PM10']
    for col in pollutant_cols:
        if col in site_data.columns:
            site_data[col] = site_data[col].fillna(site_data[col].mean())
    
    if len(site_data) < seq_length + 1:
        return sequences, targets, coords
    
    for i in range(len(site_data) - seq_length):
        seq_data = site_data.iloc[i:i+seq_length][pollutant_cols].values
        target = site_data.iloc[i+seq_length]['PM2.5']
        lat = site_data.iloc[i+seq_length]['Site Latitude']
        lon = site_data.iloc[i+seq_length]['Site Longitude']
        
        if not np.isnan(target) and not np.any(np.isnan(seq_data)):
            sequences.append(seq_data)
            targets.append(target)
            coords.append((lat, lon))
    
    return sequences, targets, coords

# Recreate the test data with coordinates
all_sequences = []
all_targets = []
all_coords = []
site_names = []

for site_name in df_clean['Local Site Name'].unique():
    site_data = df_clean[df_clean['Local Site Name'] == site_name].copy()
    sequences, targets, coords = create_sequences_with_coords(site_data, seq_length=6)
    
    if sequences:
        all_sequences.extend(sequences)
        all_targets.extend(targets)
        all_coords.extend(coords)
        site_names.extend([site_name] * len(sequences))

# Split to get test indices (same random state as training)
from sklearn.model_selection import train_test_split
X = np.array(all_sequences)
y = np.array(all_targets)
coords_array = np.array(all_coords)

X_train, X_test, y_train, y_test, coords_train, coords_test, sites_train, sites_test = train_test_split(
    X, y, coords_array, site_names, test_size=0.2, random_state=42
)

print(f'Test data: {len(y_test)} samples')

# Create DataFrame with test data and predictions
test_data = pd.DataFrame({
    'Site_Name': sites_test,
    'Latitude': coords_test[:, 0],
    'Longitude': coords_test[:, 1],
    'Actual_PM25': predictions_df['Actual_PM25'].values,
    'Predicted_PM25': predictions_df['Predicted_PM25'].values,
    'Absolute_Error': predictions_df['Absolute_Error'].values
})

# Calculate center of map
center_lat = test_data['Latitude'].mean()
center_lon = test_data['Longitude'].mean()

print(f'Map center: {center_lat:.4f}, {center_lon:.4f}')

# Create base map
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10,
    tiles='OpenStreetMap'
)

# Add title
title_html = '''
<h3 align="center" style="font-size:20px"><b>KSC-ConvLSTM Model: Actual vs Predicted PM2.5</b></h3>
<p align="center" style="font-size:14px">Chicago Area Air Quality Monitoring Sites</p>
'''
m.get_root().html.add_child(folium.Element(title_html))

# Color functions for actual and predicted values
def get_color_actual(value):
    if value < 35:
        return 'green'
    elif value < 55:
        return 'orange'
    else:
        return 'red'

def get_color_predicted(value):
    if value < 35:
        return 'lightgreen'
    elif value < 55:
        return 'yellow'
    else:
        return 'pink'

def get_error_color(error):
    if error < 5:
        return 'blue'
    elif error < 10:
        return 'purple'
    else:
        return 'darkred'

# Add markers for each test point
for idx, row in test_data.iterrows():
    # Actual PM2.5 marker
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=8,
        popup=folium.Popup(f"""
        <b>Site:</b> {row['Site_Name']}<br>
        <b>Actual PM2.5:</b> {row['Actual_PM25']:.2f} μg/m³<br>
        <b>Predicted PM2.5:</b> {row['Predicted_PM25']:.2f} μg/m³<br>
        <b>Error:</b> {row['Absolute_Error']:.2f} μg/m³<br>
        <b>Coordinates:</b> {row['Latitude']:.4f}, {row['Longitude']:.4f}
        """, max_width=300),
        color='black',
        weight=2,
        fillColor=get_color_actual(row['Actual_PM25']),
        fillOpacity=0.8,
        tooltip=f"Actual: {row['Actual_PM25']:.1f} μg/m³"
    ).add_to(m)
    
    # Predicted PM2.5 marker (slightly offset)
    folium.CircleMarker(
        location=[row['Latitude'] + 0.002, row['Longitude'] + 0.002],
        radius=6,
        popup=folium.Popup(f"""
        <b>Site:</b> {row['Site_Name']}<br>
        <b>Predicted PM2.5:</b> {row['Predicted_PM25']:.2f} μg/m³<br>
        <b>Actual PM2.5:</b> {row['Actual_PM25']:.2f} μg/m³<br>
        <b>Error:</b> {row['Absolute_Error']:.2f} μg/m³
        """, max_width=300),
        color='darkblue',
        weight=2,
        fillColor=get_color_predicted(row['Predicted_PM25']),
        fillOpacity=0.7,
        tooltip=f"Predicted: {row['Predicted_PM25']:.1f} μg/m³"
    ).add_to(m)

# Add legend
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 200px; height: 180px; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 10px">
<p><b>Legend</b></p>
<p><i class="fa fa-circle" style="color:green"></i> Actual PM2.5 < 35 μg/m³</p>
<p><i class="fa fa-circle" style="color:orange"></i> Actual PM2.5 35-55 μg/m³</p>
<p><i class="fa fa-circle" style="color:red"></i> Actual PM2.5 > 55 μg/m³</p>
<p><i class="fa fa-circle" style="color:lightgreen"></i> Predicted PM2.5 < 35 μg/m³</p>
<p><i class="fa fa-circle" style="color:yellow"></i> Predicted PM2.5 35-55 μg/m³</p>
<p><i class="fa fa-circle" style="color:pink"></i> Predicted PM2.5 > 55 μg/m³</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Add statistics box
stats_html = f'''
<div style="position: fixed; 
            top: 80px; right: 50px; width: 250px; height: 150px; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:12px; padding: 10px">
<p><b>Model Performance</b></p>
<p>Test Samples: {len(test_data)}</p>
<p>MAE: {test_data['Absolute_Error'].mean():.2f} μg/m³</p>
<p>RMSE: {np.sqrt((test_data['Absolute_Error']**2).mean()):.2f} μg/m³</p>
<p>Mean Actual: {test_data['Actual_PM25'].mean():.2f} μg/m³</p>
<p>Mean Predicted: {test_data['Predicted_PM25'].mean():.2f} μg/m³</p>
<p>Max Error: {test_data['Absolute_Error'].max():.2f} μg/m³</p>
</div>
'''
m.get_root().html.add_child(folium.Element(stats_html))

# Save the map
map_filename = 'ksc_model_comparison.html'
m.save(map_filename)
print(f'Map saved as {map_filename}')

# Create a second map showing error distribution
m2 = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10,
    tiles='OpenStreetMap'
)

# Add title for error map
title_html2 = '''
<h3 align="center" style="font-size:20px"><b>KSC-ConvLSTM Model: Prediction Error Distribution</b></h3>
<p align="center" style="font-size:14px">Absolute Error by Location</p>
'''
m2.get_root().html.add_child(folium.Element(title_html2))

# Add error markers
for idx, row in test_data.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=max(5, min(20, row['Absolute_Error'])),  # Size based on error
        popup=folium.Popup(f"""
        <b>Site:</b> {row['Site_Name']}<br>
        <b>Absolute Error:</b> {row['Absolute_Error']:.2f} μg/m³<br>
        <b>Actual PM2.5:</b> {row['Actual_PM25']:.2f} μg/m³<br>
        <b>Predicted PM2.5:</b> {row['Predicted_PM25']:.2f} μg/m³<br>
        <b>Error %:</b> {(row['Absolute_Error']/row['Actual_PM25']*100):.1f}%
        """, max_width=300),
        color='black',
        weight=1,
        fillColor=get_error_color(row['Absolute_Error']),
        fillOpacity=0.7,
        tooltip=f"Error: {row['Absolute_Error']:.1f} μg/m³"
    ).add_to(m2)

# Add error legend
error_legend_html = '''
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 200px; height: 120px; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 10px">
<p><b>Error Legend</b></p>
<p><i class="fa fa-circle" style="color:blue"></i> Error < 5 μg/m³</p>
<p><i class="fa fa-circle" style="color:purple"></i> Error 5-10 μg/m³</p>
<p><i class="fa fa-circle" style="color:darkred"></i> Error > 10 μg/m³</p>
<p><small>Circle size = Error magnitude</small></p>
</div>
'''
m2.get_root().html.add_child(folium.Element(error_legend_html))

# Save the error map
error_map_filename = 'ksc_model_comparison_errors.html'
m2.save(error_map_filename)
print(f'Error map saved as {error_map_filename}')

print('\n=== Map Creation Complete ===')
print(f'Created two maps:')
print(f'1. {map_filename} - Actual vs Predicted comparison')
print(f'2. {error_map_filename} - Error distribution')
