import pandas as pd
import numpy as np
import folium
from folium import plugins
import warnings
warnings.filterwarnings('ignore')

print('=== Creating KSC Side-by-Side Comparison Map ===')

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

# Color function for PM2.5 values
def get_pm25_color(value):
    if value < 12:
        return 'green'      # Good
    elif value < 35.5:
        return 'yellow'     # Moderate
    elif value < 55.5:
        return 'orange'     # Unhealthy for Sensitive Groups
    elif value < 150.5:
        return 'red'        # Unhealthy
    else:
        return 'purple'     # Very Unhealthy

def get_marker_size(value):
    # Scale marker size based on PM2.5 value
    return max(8, min(20, value * 0.3))

# Create the main HTML structure with side-by-side maps
html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>KSC Model: Actual vs Predicted PM2.5 Comparison</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            text-align: center;
            padding: 15px;
        }}
        .container {{
            display: flex;
            height: calc(100vh - 80px);
        }}
        .map-container {{
            flex: 1;
            position: relative;
        }}
        .map {{
            height: 100%;
            width: 100%;
        }}
        .map-title {{
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(255,255,255,0.9);
            padding: 10px 20px;
            border-radius: 5px;
            z-index: 1000;
            font-weight: bold;
            font-size: 16px;
        }}
        .legend {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(255,255,255,0.9);
            padding: 15px;
            border-radius: 5px;
            z-index: 1000;
            font-size: 12px;
            line-height: 1.4;
        }}
        .stats {{
            position: absolute;
            top: 60px;
            right: 20px;
            background: rgba(255,255,255,0.9);
            padding: 15px;
            border-radius: 5px;
            z-index: 1000;
            font-size: 12px;
            line-height: 1.4;
            min-width: 200px;
        }}
        .divider {{
            width: 2px;
            background-color: #34495e;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>KSC-ConvLSTM Model: Actual vs Predicted PM2.5 Comparison</h1>
        <p>Chicago Area Air Quality Monitoring Sites - Test Data Analysis</p>
    </div>
    <div class="container">
        <div class="map-container">
            <div class="map-title">Actual PM2.5 Values</div>
            <div id="actual-map" class="map"></div>
            <div class="legend">
                <strong>PM2.5 AQI Levels</strong><br>
                <span style="color: green;">●</span> Good (0-12 μg/m³)<br>
                <span style="color: yellow;">●</span> Moderate (12-35.5 μg/m³)<br>
                <span style="color: orange;">●</span> Unhealthy for Sensitive (35.5-55.5 μg/m³)<br>
                <span style="color: red;">●</span> Unhealthy (55.5-150.5 μg/m³)<br>
                <span style="color: purple;">●</span> Very Unhealthy (>150.5 μg/m³)
            </div>
        </div>
        <div class="divider"></div>
        <div class="map-container">
            <div class="map-title">Predicted PM2.5 Values</div>
            <div id="predicted-map" class="map"></div>
            <div class="stats">
                <strong>Model Performance</strong><br>
                Test Samples: {test_samples}<br>
                MAE: {mae:.2f} μg/m³<br>
                RMSE: {rmse:.2f} μg/m³<br>
                Mean Actual: {mean_actual:.2f} μg/m³<br>
                Mean Predicted: {mean_predicted:.2f} μg/m³<br>
                Max Error: {max_error:.2f} μg/m³<br>
                R² Score: {r2_score:.3f}
            </div>
        </div>
    </div>

    <script>
        // Initialize maps
        var actualMap = L.map('actual-map').setView([{center_lat}, {center_lon}], 10);
        var predictedMap = L.map('predicted-map').setView([{center_lat}, {center_lon}], 10);

        // Add tile layers
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(actualMap);

        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(predictedMap);

        // Synchronize map movements
        actualMap.on('moveend', function() {{
            predictedMap.setView(actualMap.getCenter(), actualMap.getZoom());
        }});

        predictedMap.on('moveend', function() {{
            actualMap.setView(predictedMap.getCenter(), predictedMap.getZoom());
        }});

        // Function to get color based on PM2.5 value
        function getPM25Color(value) {{
            if (value < 12) return 'green';
            else if (value < 35.5) return 'yellow';
            else if (value < 55.5) return 'orange';
            else if (value < 150.5) return 'red';
            else return 'purple';
        }}

        // Function to get marker size
        function getMarkerSize(value) {{
            return Math.max(8, Math.min(20, value * 0.3));
        }}

        // Add markers for actual values
        {actual_markers}

        // Add markers for predicted values
        {predicted_markers}
    </script>
</body>
</html>
'''

# Generate marker JavaScript for actual values
actual_markers_js = []
for idx, row in test_data.iterrows():
    marker_js = f'''
        L.circleMarker([{row['Latitude']}, {row['Longitude']}], {{
            radius: {get_marker_size(row['Actual_PM25'])},
            fillColor: '{get_pm25_color(row['Actual_PM25'])}',
            color: 'black',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        }}).addTo(actualMap)
        .bindPopup(`
            <b>Site:</b> {row['Site_Name']}<br>
            <b>Actual PM2.5:</b> {row['Actual_PM25']:.2f} μg/m³<br>
            <b>Predicted PM2.5:</b> {row['Predicted_PM25']:.2f} μg/m³<br>
            <b>Absolute Error:</b> {row['Absolute_Error']:.2f} μg/m³<br>
            <b>Error %:</b> {(row['Absolute_Error']/row['Actual_PM25']*100):.1f}%<br>
            <b>Coordinates:</b> {row['Latitude']:.4f}, {row['Longitude']:.4f}
        `);'''
    actual_markers_js.append(marker_js)

# Generate marker JavaScript for predicted values
predicted_markers_js = []
for idx, row in test_data.iterrows():
    marker_js = f'''
        L.circleMarker([{row['Latitude']}, {row['Longitude']}], {{
            radius: {get_marker_size(row['Predicted_PM25'])},
            fillColor: '{get_pm25_color(row['Predicted_PM25'])}',
            color: 'black',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        }}).addTo(predictedMap)
        .bindPopup(`
            <b>Site:</b> {row['Site_Name']}<br>
            <b>Predicted PM2.5:</b> {row['Predicted_PM25']:.2f} μg/m³<br>
            <b>Actual PM2.5:</b> {row['Actual_PM25']:.2f} μg/m³<br>
            <b>Absolute Error:</b> {row['Absolute_Error']:.2f} μg/m³<br>
            <b>Error %:</b> {(row['Absolute_Error']/row['Actual_PM25']*100):.1f}%<br>
            <b>Coordinates:</b> {row['Latitude']:.4f}, {row['Longitude']:.4f}
        `);'''
    predicted_markers_js.append(marker_js)

# Calculate statistics
mae = test_data['Absolute_Error'].mean()
rmse = np.sqrt((test_data['Absolute_Error']**2).mean())
mean_actual = test_data['Actual_PM25'].mean()
mean_predicted = test_data['Predicted_PM25'].mean()
max_error = test_data['Absolute_Error'].max()

# Calculate R² score
from sklearn.metrics import r2_score
r2 = r2_score(test_data['Actual_PM25'], test_data['Predicted_PM25'])

# Fill in the template
html_content = html_template.format(
    center_lat=center_lat,
    center_lon=center_lon,
    test_samples=len(test_data),
    mae=mae,
    rmse=rmse,
    mean_actual=mean_actual,
    mean_predicted=mean_predicted,
    max_error=max_error,
    r2_score=r2,
    actual_markers='\n'.join(actual_markers_js),
    predicted_markers='\n'.join(predicted_markers_js)
)

# Save the HTML file
filename = 'ksc_take_2.html'
with open(filename, 'w') as f:
    f.write(html_content)

print(f'\nSide-by-side comparison map saved as {filename}')
print(f'Map features:')
print(f'- Left map: Actual PM2.5 values from test data')
print(f'- Right map: Predicted PM2.5 values from KSC model')
print(f'- Synchronized navigation between maps')
print(f'- Color-coded markers based on AQI levels')
print(f'- Marker size proportional to PM2.5 concentration')
print(f'- Interactive popups with detailed information')
print(f'- Performance statistics panel')
print(f'- AQI legend for interpretation')

print('\n=== Side-by-Side Map Creation Complete ===')
