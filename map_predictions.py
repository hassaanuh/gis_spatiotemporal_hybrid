import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib
import folium
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Custom objects for model loading
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

custom_objects = {
    'mse': mse,
    'MeanSquaredError': MeanSquaredError
}

class ChicagoAirQualityMapper:
    def __init__(self, model_path, scaler_path, data_path):
        """
        Initialize the mapper with trained model, scaler, and data
        """
        self.model = load_model(model_path, custom_objects=custom_objects)
        self.scaler = joblib.load(scaler_path)
        self.data = pd.read_csv(data_path, index_col=[0, 1, 2, 3])
        self.grid_h = 10
        self.grid_w = 10
        self.seq_length = 6
        
        # Chicago center coordinates
        self.chicago_center = [41.8781, -87.6298]
        
        print("Model and data loaded successfully!")
        print(f"Model input shape: {self.model.input_shape}")
        print(f"Model output shape: {self.model.output_shape}")
        
    def prepare_test_data(self, prediction_horizon=24):
        """
        Prepare test data for mapping predictions vs actual values
        """
        print(f"Preparing test data for {prediction_horizon}h predictions...")
        
        # Reset index to work with the data
        df_reset = self.data.reset_index()
        
        # Convert Month_Year to datetime
        df_reset['Month_Year'] = pd.to_datetime(df_reset['Month_Year'])
        
        # Sort by location and time
        df_reset = df_reset.sort_values(['Local Site Name', 'Month_Year'])
        
        # Define pollutant columns
        pollutant_cols = ['CO', 'NO', 'PM10', 'PM2.5', 'SO2']
        
        # Fill missing values
        for col in pollutant_cols:
            df_reset[col] = df_reset.groupby('Local Site Name')[col].fillna(method='ffill')
            df_reset[col] = df_reset.groupby('Local Site Name')[col].fillna(method='bfill')
            df_reset[col] = df_reset[col].fillna(df_reset[col].median())
        
        # Remove any remaining rows with NaN values
        df_reset = df_reset.dropna(subset=pollutant_cols)
        
        # Store original values before scaling for display
        df_original = df_reset.copy()
        
        # Normalize the data using the saved scaler
        df_reset[pollutant_cols] = self.scaler.transform(df_reset[pollutant_cols])
        
        return df_reset, df_original
    
    def create_test_sequences(self, df_reset, df_original):
        """
        Create test sequences and get predictions
        """
        sequences = []
        targets = []
        locations = []
        dates = []
        coordinates = []
        original_targets = []
        
        # Group by location
        for location, group in df_reset.groupby('Local Site Name'):
            if len(group) < self.seq_length + 1:
                continue
                
            # Sort by time
            group = group.sort_values('Month_Year')
            original_group = df_original[df_original['Local Site Name'] == location].sort_values('Month_Year')
            
            # Use the last available sequence for prediction (most recent data)
            if len(group) >= self.seq_length + 1:
                # Input sequence (last seq_length timesteps)
                seq_data = group.iloc[-self.seq_length-1:-1][['CO', 'NO', 'PM10', 'PM2.5', 'SO2']].values
                
                # Target (most recent PM2.5)
                target = group.iloc[-1]['PM2.5']
                original_target = original_group.iloc[-1]['PM2.5']
                
                # Get coordinates
                lat = group.iloc[-1]['Site Latitude']
                lon = group.iloc[-1]['Site Longitude']
                
                sequences.append(seq_data)
                targets.append(target)
                original_targets.append(original_target)
                locations.append(location)
                dates.append(group.iloc[-1]['Month_Year'])
                coordinates.append((lat, lon))
        
        if len(sequences) == 0:
            print("No sequences could be created!")
            return None
        
        # Convert to numpy arrays
        X = np.array(sequences)
        y_true = np.array(targets)
        y_true_original = np.array(original_targets)
        
        # Reshape X to match model input
        n_samples, seq_len, n_features = X.shape
        X_spatial = np.zeros((n_samples, seq_len, self.grid_h, self.grid_w, n_features))
        
        # Fill the center of each grid with the actual data
        center_h, center_w = self.grid_h // 2, self.grid_w // 2
        X_spatial[:, :, center_h, center_w, :] = X
        
        # Add some spatial variation
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                h_idx = center_h + i
                w_idx = center_w + j
                if 0 <= h_idx < self.grid_h and 0 <= w_idx < self.grid_w:
                    noise = np.random.normal(0, 0.02, X.shape)
                    X_spatial[:, :, h_idx, w_idx, :] = X + noise
        
        # Make predictions
        print("Making predictions...")
        y_pred_spatial = self.model.predict(X_spatial, verbose=0)
        y_pred = y_pred_spatial[:, center_h, center_w, 0]
        
        # Inverse transform predictions to original scale
        # Create dummy array for inverse transform
        dummy_data = np.zeros((len(y_pred), 5))
        dummy_data[:, 3] = y_pred  # PM2.5 is at index 3
        y_pred_original = self.scaler.inverse_transform(dummy_data)[:, 3]
        
        return {
            'locations': locations,
            'coordinates': coordinates,
            'dates': dates,
            'y_true': y_true_original,
            'y_pred': y_pred_original,
            'y_true_scaled': y_true,
            'y_pred_scaled': y_pred
        }
    
    def create_comparison_maps(self, test_results):
        """
        Create side-by-side maps showing predicted vs actual PM2.5 values
        """
        print("Creating comparison maps...")
        
        # Get data
        locations = test_results['locations']
        coordinates = test_results['coordinates']
        y_true = test_results['y_true']
        y_pred = test_results['y_pred']
        dates = test_results['dates']
        
        # Calculate min and max values for consistent color scaling
        all_values = np.concatenate([y_true, y_pred])
        vmin, vmax = np.min(all_values), np.max(all_values)
        
        # Create colormap
        colormap = cm.get_cmap('RdYlBu_r')  # Red-Yellow-Blue reversed (red=high, blue=low)
        normalize = colors.Normalize(vmin=vmin, vmax=vmax)
        
        # Create base maps
        map_actual = folium.Map(
            location=self.chicago_center,
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        map_predicted = folium.Map(
            location=self.chicago_center,
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Add markers for actual values
        for i, (location, coord, actual, predicted, date) in enumerate(zip(locations, coordinates, y_true, y_pred, dates)):
            lat, lon = coord
            
            # Color for actual values
            actual_color = colors.to_hex(colormap(normalize(actual)))
            
            # Color for predicted values
            pred_color = colors.to_hex(colormap(normalize(predicted)))
            
            # Actual values map
            folium.CircleMarker(
                location=[lat, lon],
                radius=12,
                popup=folium.Popup(
                    f"""
                    <b>{location}</b><br>
                    <b>Actual PM2.5:</b> {actual:.2f} μg/m³<br>
                    <b>Date:</b> {date.strftime('%Y-%m')}<br>
                    <b>Coordinates:</b> {lat:.4f}, {lon:.4f}
                    """,
                    max_width=300
                ),
                tooltip=f"{location}: {actual:.2f} μg/m³",
                color='black',
                weight=2,
                fillColor=actual_color,
                fillOpacity=0.8
            ).add_to(map_actual)
            
            # Predicted values map
            folium.CircleMarker(
                location=[lat, lon],
                radius=12,
                popup=folium.Popup(
                    f"""
                    <b>{location}</b><br>
                    <b>Predicted PM2.5:</b> {predicted:.2f} μg/m³<br>
                    <b>Actual PM2.5:</b> {actual:.2f} μg/m³<br>
                    <b>Error:</b> {abs(predicted - actual):.2f} μg/m³<br>
                    <b>Date:</b> {date.strftime('%Y-%m')}<br>
                    <b>Coordinates:</b> {lat:.4f}, {lon:.4f}
                    """,
                    max_width=300
                ),
                tooltip=f"{location}: {predicted:.2f} μg/m³ (pred)",
                color='black',
                weight=2,
                fillColor=pred_color,
                fillOpacity=0.8
            ).add_to(map_predicted)
        
        # Add legends
        self._add_colorbar_legend(map_actual, vmin, vmax, "Actual PM2.5 (μg/m³)")
        self._add_colorbar_legend(map_predicted, vmin, vmax, "Predicted PM2.5 (μg/m³)")
        
        # Add titles
        title_actual = """
        <h3 align="center" style="font-size:20px"><b>Actual PM2.5 Concentrations in Chicago</b></h3>
        """
        title_predicted = """
        <h3 align="center" style="font-size:20px"><b>Predicted PM2.5 Concentrations in Chicago</b></h3>
        """
        
        map_actual.get_root().html.add_child(folium.Element(title_actual))
        map_predicted.get_root().html.add_child(folium.Element(title_predicted))
        
        # Save maps
        map_actual.save('chicago_actual_pm25_map.html')
        map_predicted.save('chicago_predicted_pm25_map.html')
        
        print("Maps saved as:")
        print("- chicago_actual_pm25_map.html")
        print("- chicago_predicted_pm25_map.html")
        
        return map_actual, map_predicted
    
    def _add_colorbar_legend(self, map_obj, vmin, vmax, title):
        """
        Add a colorbar legend to the map
        """
        # Create colorbar HTML
        colorbar_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 150px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p style="margin: 0; font-weight: bold;">{title}</p>
        <div style="background: linear-gradient(to top, 
                    #313695 0%, #4575b4 25%, #74add1 50%, 
                    #abd9e9 75%, #fee090 85%, #fdae61 95%, #d73027 100%);
                    width: 20px; height: 50px; float: left; margin-right: 10px;"></div>
        <div style="float: left;">
            <div style="height: 12px; line-height: 12px;">{vmax:.1f}</div>
            <div style="height: 26px;"></div>
            <div style="height: 12px; line-height: 12px;">{vmin:.1f}</div>
        </div>
        </div>
        '''
        
        map_obj.get_root().html.add_child(folium.Element(colorbar_html))
    
    def create_side_by_side_html(self):
        """
        Create a single HTML file with both maps side by side
        """
        print("Creating side-by-side comparison...")
        
        side_by_side_html = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Chicago PM2.5 Predictions vs Actual Values</title>
            <style>
                body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
                .container { display: flex; width: 100%; height: 80vh; gap: 20px; }
                .map-container { flex: 1; border: 2px solid #ccc; border-radius: 8px; }
                .map-container iframe { width: 100%; height: 100%; border: none; border-radius: 6px; }
                .header { text-align: center; margin-bottom: 20px; }
                .stats { margin-top: 20px; padding: 15px; background-color: #f5f5f5; border-radius: 8px; }
                .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
                .stat-item { background: white; padding: 10px; border-radius: 5px; text-align: center; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Chicago Air Quality Model: Predictions vs Actual Values</h1>
                <p>Interactive comparison of predicted and actual PM2.5 concentrations across Chicago monitoring stations</p>
            </div>
            
            <div class="container">
                <div class="map-container">
                    <iframe src="chicago_actual_pm25_map.html"></iframe>
                </div>
                <div class="map-container">
                    <iframe src="chicago_predicted_pm25_map.html"></iframe>
                </div>
            </div>
            
            <div class="stats">
                <h3>Model Performance Summary</h3>
                <div class="stats-grid" id="stats-content">
                    <!-- Stats will be populated by JavaScript -->
                </div>
            </div>
            
            <script>
                // This would be populated with actual statistics
                document.getElementById('stats-content').innerHTML = `
                    <div class="stat-item">
                        <h4>Mean Absolute Error</h4>
                        <p id="mae">Loading...</p>
                    </div>
                    <div class="stat-item">
                        <h4>Root Mean Square Error</h4>
                        <p id="rmse">Loading...</p>
                    </div>
                    <div class="stat-item">
                        <h4>R² Score</h4>
                        <p id="r2">Loading...</p>
                    </div>
                    <div class="stat-item">
                        <h4>Number of Locations</h4>
                        <p id="n_locations">Loading...</p>
                    </div>
                `;
            </script>
        </body>
        </html>
        '''
        
        with open('chicago_pm25_comparison.html', 'w') as f:
            f.write(side_by_side_html)
        
        print("Side-by-side comparison saved as: chicago_pm25_comparison.html")
    
    def calculate_and_display_metrics(self, test_results):
        """
        Calculate and display performance metrics
        """
        y_true = test_results['y_true']
        y_pred = test_results['y_pred']
        locations = test_results['locations']
        
        # Overall metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print("\n" + "="*60)
        print("CHICAGO AIR QUALITY MAPPING RESULTS")
        print("="*60)
        print(f"Number of monitoring locations: {len(locations)}")
        print(f"Mean Absolute Error (MAE): {mae:.3f} μg/m³")
        print(f"Root Mean Square Error (RMSE): {rmse:.3f} μg/m³")
        print(f"R² Score: {r2:.3f}")
        
        # Location-wise performance
        print(f"\nLocation-wise Performance:")
        print("-" * 40)
        
        location_errors = []
        for i, location in enumerate(locations):
            error = abs(y_pred[i] - y_true[i])
            location_errors.append((location, error, y_true[i], y_pred[i]))
        
        # Sort by error
        location_errors.sort(key=lambda x: x[1])
        
        print("Best Predictions (lowest error):")
        for i, (location, error, actual, pred) in enumerate(location_errors[:5]):
            print(f"  {i+1}. {location[:40]}")
            print(f"     Actual: {actual:.2f}, Predicted: {pred:.2f}, Error: {error:.2f} μg/m³")
        
        print("\nWorst Predictions (highest error):")
        for i, (location, error, actual, pred) in enumerate(location_errors[-5:]):
            print(f"  {i+1}. {location[:40]}")
            print(f"     Actual: {actual:.2f}, Predicted: {pred:.2f}, Error: {error:.2f} μg/m³")
        
        return mae, rmse, r2

def main():
    """
    Main function to create Chicago air quality prediction maps
    """
    print("Starting Chicago Air Quality Mapping")
    print("="*50)
    
    # Initialize the mapper
    mapper = ChicagoAirQualityMapper(
        model_path='simplified_convlstm_air_quality_model.h5',
        scaler_path='air_quality_scaler_fixed.pkl',
        data_path='grid_data/combined.csv'
    )
    
    # Prepare test data
    df_scaled, df_original = mapper.prepare_test_data(prediction_horizon=24)
    
    # Create test sequences and get predictions
    test_results = mapper.create_test_sequences(df_scaled, df_original)
    
    if test_results is None:
        print("Failed to create test sequences. Exiting.")
        return
    
    # Calculate and display metrics
    mae, rmse, r2 = mapper.calculate_and_display_metrics(test_results)
    
    # Create comparison maps
    map_actual, map_predicted = mapper.create_comparison_maps(test_results)
    
    # Create side-by-side HTML
    mapper.create_side_by_side_html()
    
    print("\n" + "="*60)
    print("MAPPING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Generated files:")
    print("1. chicago_actual_pm25_map.html - Interactive map of actual PM2.5 values")
    print("2. chicago_predicted_pm25_map.html - Interactive map of predicted PM2.5 values")
    print("3. chicago_pm25_comparison.html - Side-by-side comparison view")
    print("\nOpen chicago_pm25_comparison.html in your web browser to view the comparison!")

if __name__ == "__main__":
    main()
