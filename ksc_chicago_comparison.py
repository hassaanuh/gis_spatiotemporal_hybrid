import pandas as pd
import numpy as np
import folium
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class KSCChicagoMapper:
    def __init__(self):
        """
        Initialize the KSC Chicago mapper
        """
        # Chicago center coordinates
        self.chicago_center = [41.8781, -87.6298]
        
        print("KSC Chicago Air Quality Mapper initialized!")
        
    def load_ksc_predictions(self):
        """
        Load KSC model predictions and prepare data with coordinates
        """
        print("Loading KSC model predictions...")
        
        # Load the prediction results
        try:
            predictions_df = pd.read_csv('ksc_model_predictions.csv')
            print(f"Loaded {len(predictions_df)} prediction records")
        except FileNotFoundError:
            print("KSC predictions file not found. Please run the KSC model first.")
            return None
        
        # Load the original data to get coordinates and recreate the test split
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
        
        # Recreate the exact same sequences as in training
        def create_sequences_with_coords(site_data, seq_length=6):
            sequences = []
            targets = []
            coords = []
            site_names = []
            dates = []
            
            site_data = site_data.sort_values('Month_Year')
            
            pollutant_cols = ['PM2.5', 'CO', 'SO2', 'NO', 'PM10']
            for col in pollutant_cols:
                if col in site_data.columns:
                    site_data[col] = site_data[col].fillna(site_data[col].mean())
            
            if len(site_data) < seq_length + 1:
                return sequences, targets, coords, site_names, dates
            
            for i in range(len(site_data) - seq_length):
                seq_data = site_data.iloc[i:i+seq_length][pollutant_cols].values
                target = site_data.iloc[i+seq_length]['PM2.5']
                lat = site_data.iloc[i+seq_length]['Site Latitude']
                lon = site_data.iloc[i+seq_length]['Site Longitude']
                site_name = site_data.iloc[i+seq_length]['Local Site Name']
                date = site_data.iloc[i+seq_length]['Month_Year']
                
                if not np.isnan(target) and not np.any(np.isnan(seq_data)):
                    sequences.append(seq_data)
                    targets.append(target)
                    coords.append((lat, lon))
                    site_names.append(site_name)
                    dates.append(date)
            
            return sequences, targets, coords, site_names, dates
        
        # Recreate all sequences
        all_sequences = []
        all_targets = []
        all_coords = []
        all_site_names = []
        all_dates = []
        
        for site_name in df_clean['Local Site Name'].unique():
            site_data = df_clean[df_clean['Local Site Name'] == site_name].copy()
            sequences, targets, coords, site_names, dates = create_sequences_with_coords(site_data, seq_length=6)
            
            if sequences:
                all_sequences.extend(sequences)
                all_targets.extend(targets)
                all_coords.extend(coords)
                all_site_names.extend(site_names)
                all_dates.extend(dates)
        
        # Split to get test indices (same random state as training)
        from sklearn.model_selection import train_test_split
        X = np.array(all_sequences)
        y = np.array(all_targets)
        coords_array = np.array(all_coords)
        
        X_train, X_test, y_train, y_test, coords_train, coords_test, sites_train, sites_test, dates_train, dates_test = train_test_split(
            X, y, coords_array, all_site_names, all_dates, test_size=0.2, random_state=42
        )
        
        print(f"Test set size: {len(y_test)}")
        print(f"Predictions size: {len(predictions_df)}")
        
        # Match the predictions with test data
        if len(predictions_df) != len(y_test):
            print(f"Warning: Prediction count ({len(predictions_df)}) doesn't match test count ({len(y_test)})")
            # Take the minimum to avoid index errors
            min_len = min(len(predictions_df), len(y_test))
            predictions_df = predictions_df.iloc[:min_len]
            y_test = y_test[:min_len]
            coords_test = coords_test[:min_len]
            sites_test = sites_test[:min_len]
            dates_test = dates_test[:min_len]
        
        # Create test results dictionary
        test_results = {
            'locations': sites_test,
            'coordinates': coords_test,
            'dates': dates_test,
            'y_true': predictions_df['Actual_PM25'].values,
            'y_pred': predictions_df['Predicted_PM25'].values,
            'errors': predictions_df['Absolute_Error'].values
        }
        
        print(f"Prepared test data with {len(test_results['locations'])} locations")
        return test_results
    
    def create_comparison_maps(self, test_results):
        """
        Create side-by-side maps showing predicted vs actual PM2.5 values
        """
        print("Creating KSC comparison maps...")
        
        # Get data
        locations = test_results['locations']
        coordinates = test_results['coordinates']
        y_true = test_results['y_true']
        y_pred = test_results['y_pred']
        dates = test_results['dates']
        errors = test_results['errors']
        
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
        for i, (location, coord, actual, predicted, date, error) in enumerate(zip(locations, coordinates, y_true, y_pred, dates, errors)):
            lat, lon = coord
            
            # Color for actual values
            actual_color = colors.to_hex(colormap(normalize(actual)))
            
            # Color for predicted values
            pred_color = colors.to_hex(colormap(normalize(predicted)))
            
            # Format date
            date_str = date.strftime('%Y-%m') if hasattr(date, 'strftime') else str(date)
            
            # Actual values map
            folium.CircleMarker(
                location=[lat, lon],
                radius=15,
                popup=folium.Popup(
                    f"""
                    <div style="font-family: Arial, sans-serif; width: 280px;">
                        <h4 style="margin: 0 0 10px 0; color: #2c3e50; text-align: center;">{location}</h4>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr style="background-color: #e8f5e8;">
                                <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Actual PM2.5:</td>
                                <td style="padding: 8px; border: 1px solid #dee2e6; color: #e74c3c; font-weight: bold;">{actual:.2f} μg/m³</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Predicted PM2.5:</td>
                                <td style="padding: 8px; border: 1px solid #dee2e6;">{predicted:.2f} μg/m³</td>
                            </tr>
                            <tr style="background-color: #fff3cd;">
                                <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Absolute Error:</td>
                                <td style="padding: 8px; border: 1px solid #dee2e6;">{error:.2f} μg/m³</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Error %:</td>
                                <td style="padding: 8px; border: 1px solid #dee2e6;">{(error/actual*100):.1f}%</td>
                            </tr>
                            <tr style="background-color: #f8f9fa;">
                                <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Date:</td>
                                <td style="padding: 8px; border: 1px solid #dee2e6;">{date_str}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Coordinates:</td>
                                <td style="padding: 8px; border: 1px solid #dee2e6;">{lat:.4f}, {lon:.4f}</td>
                            </tr>
                        </table>
                    </div>
                    """,
                    max_width=320
                ),
                tooltip=f"{location}: {actual:.2f} μg/m³ (actual)",
                color='black',
                weight=2,
                fillColor=actual_color,
                fillOpacity=0.8
            ).add_to(map_actual)
            
            # Predicted values map
            folium.CircleMarker(
                location=[lat, lon],
                radius=15,
                popup=folium.Popup(
                    f"""
                    <div style="font-family: Arial, sans-serif; width: 280px;">
                        <h4 style="margin: 0 0 10px 0; color: #2c3e50; text-align: center;">{location}</h4>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr style="background-color: #e3f2fd;">
                                <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Predicted PM2.5:</td>
                                <td style="padding: 8px; border: 1px solid #dee2e6; color: #3498db; font-weight: bold;">{predicted:.2f} μg/m³</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Actual PM2.5:</td>
                                <td style="padding: 8px; border: 1px solid #dee2e6;">{actual:.2f} μg/m³</td>
                            </tr>
                            <tr style="background-color: #fff3cd;">
                                <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Absolute Error:</td>
                                <td style="padding: 8px; border: 1px solid #dee2e6;">{error:.2f} μg/m³</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Error %:</td>
                                <td style="padding: 8px; border: 1px solid #dee2e6;">{(error/actual*100):.1f}%</td>
                            </tr>
                            <tr style="background-color: #f8f9fa;">
                                <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Date:</td>
                                <td style="padding: 8px; border: 1px solid #dee2e6;">{date_str}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Coordinates:</td>
                                <td style="padding: 8px; border: 1px solid #dee2e6;">{lat:.4f}, {lon:.4f}</td>
                            </tr>
                        </table>
                    </div>
                    """,
                    max_width=320
                ),
                tooltip=f"{location}: {predicted:.2f} μg/m³ (predicted)",
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
        <h3 align="center" style="font-size:22px; color: #2c3e50; margin: 15px; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);"><b>KSC Model: Actual PM2.5 Concentrations</b></h3>
        """
        title_predicted = """
        <h3 align="center" style="font-size:22px; color: #2c3e50; margin: 15px; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);"><b>KSC Model: Predicted PM2.5 Concentrations</b></h3>
        """
        
        map_actual.get_root().html.add_child(folium.Element(title_actual))
        map_predicted.get_root().html.add_child(folium.Element(title_predicted))
        
        # Save maps
        map_actual.save('gis-maps/ksc_chicago_actual_pm25_map.html')
        map_predicted.save('gis-maps/ksc_chicago_predicted_pm25_map.html')
        
        print("Maps saved as:")
        print("- gis-maps/ksc_chicago_actual_pm25_map.html")
        print("- gis-maps/ksc_chicago_predicted_pm25_map.html")
        
        return map_actual, map_predicted
    
    def _add_colorbar_legend(self, map_obj, vmin, vmax, title):
        """
        Add a colorbar legend to the map
        """
        # Create colorbar HTML
        colorbar_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 170px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 12px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        <p style="margin: 0 0 8px 0; font-weight: bold; text-align: center; color: #2c3e50;">{title}</p>
        <div style="background: linear-gradient(to top, 
                    #313695 0%, #4575b4 25%, #74add1 50%, 
                    #abd9e9 75%, #fee090 85%, #fdae61 95%, #d73027 100%);
                    width: 30px; height: 70px; float: left; margin-right: 12px; border: 1px solid #ccc; border-radius: 3px;"></div>
        <div style="float: left; height: 70px; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 11px; font-weight: bold; color: #2c3e50;">{vmax:.1f}</div>
            <div style="font-size: 11px; font-weight: bold; color: #2c3e50;">{(vmax+vmin)/2:.1f}</div>
            <div style="font-size: 11px; font-weight: bold; color: #2c3e50;">{vmin:.1f}</div>
        </div>
        <div style="clear: both; margin-top: 8px; font-size: 10px; text-align: center; color: #666; font-style: italic;">
            KSC-ConvLSTM Model
        </div>
        </div>
        '''
        
        map_obj.get_root().html.add_child(folium.Element(colorbar_html))
    
    def create_side_by_side_html(self, test_results):
        """
        Create a single HTML file with both maps side by side
        """
        print("Creating KSC side-by-side comparison...")
        
        # Calculate metrics
        y_true = test_results['y_true']
        y_pred = test_results['y_pred']
        locations = test_results['locations']
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        side_by_side_html = f'''<!DOCTYPE html>
<html>
<head>
    <title>KSC Model: Chicago PM2.5 Predictions vs Actual Values</title>
    <style>
        body {{ 
            margin: 0; 
            padding: 20px; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }}
        .container {{ 
            display: flex; 
            width: 100%; 
            height: 75vh; 
            gap: 20px; 
            margin-bottom: 20px;
        }}
        .map-container {{ 
            flex: 1; 
            border: 3px solid #2c3e50; 
            border-radius: 15px; 
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
            overflow: hidden;
            background: white;
        }}
        .map-container iframe {{ 
            width: 100%; 
            height: 100%; 
            border: none; 
        }}
        .header {{ 
            text-align: center; 
            margin-bottom: 30px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        }}
        .header h1 {{
            margin: 0 0 15px 0;
            font-size: 2.5em;
            font-weight: 300;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .header p {{
            margin: 0;
            font-size: 1.2em;
            opacity: 0.95;
        }}
        .stats {{ 
            margin-top: 20px; 
            padding: 30px; 
            background: white;
            border-radius: 15px; 
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        }}
        .stats h3 {{
            margin: 0 0 25px 0;
            color: #2c3e50;
            font-size: 1.8em;
            text-align: center;
            font-weight: 300;
        }}
        .stats-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); 
            gap: 25px; 
        }}
        .stat-item {{ 
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 25px; 
            border-radius: 12px; 
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            transition: transform 0.3s ease;
        }}
        .stat-item:hover {{
            transform: translateY(-5px);
        }}
        .stat-item h4 {{
            margin: 0 0 15px 0;
            font-size: 1.2em;
            font-weight: 500;
            opacity: 0.9;
        }}
        .stat-item p {{
            margin: 0;
            font-size: 2.2em;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }}
        .model-info {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-top: 20px;
            text-align: center;
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        }}
        .model-info h4 {{
            margin: 0 0 15px 0;
            font-size: 1.5em;
            font-weight: 300;
        }}
        .model-info p {{
            margin: 8px 0;
            opacity: 0.95;
            font-size: 1.05em;
        }}
        .model-info strong {{
            color: #fff;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>KSC-ConvLSTM Model: Chicago Air Quality Analysis</h1>
        <p>Interactive comparison of predicted and actual PM2.5 concentrations across Chicago monitoring stations</p>
    </div>
    
    <div class="container">
        <div class="map-container">
            <iframe src="ksc_chicago_actual_pm25_map.html"></iframe>
        </div>
        <div class="map-container">
            <iframe src="ksc_chicago_predicted_pm25_map.html"></iframe>
        </div>
    </div>
    
    <div class="stats">
        <h3>KSC Model Performance Summary</h3>
        <div class="stats-grid">
            <div class="stat-item">
                <h4>Mean Absolute Error</h4>
                <p>{mae:.2f} μg/m³</p>
            </div>
            <div class="stat-item">
                <h4>Root Mean Square Error</h4>
                <p>{rmse:.2f} μg/m³</p>
            </div>
            <div class="stat-item">
                <h4>R² Score</h4>
                <p>{r2:.3f}</p>
            </div>
            <div class="stat-item">
                <h4>Test Samples</h4>
                <p>{len(y_true)}</p>
            </div>
        </div>
    </div>
    
    <div class="model-info">
        <h4>About the KSC-ConvLSTM Model</h4>
        <p><strong>Architecture:</strong> K-Nearest Neighbors Spatial Convolution + Convolutional LSTM</p>
        <p><strong>Input Features:</strong> PM2.5, CO, SO2, NO, PM10 (5 pollutants)</p>
        <p><strong>Temporal Sequence:</strong> 6-month historical data for prediction</p>
        <p><strong>Spatial Processing:</strong> Neighborhood-aware convolution with LSTM temporal modeling</p>
        <p><strong>Training Data:</strong> Chicago area air quality monitoring network (2011-2024)</p>
        <p><strong>Model Type:</strong> Spatiotemporal deep learning for air quality forecasting</p>
    </div>
</body>
</html>'''
        
        with open('gis-maps/ksc_chicago_pm25_comparison.html', 'w') as f:
            f.write(side_by_side_html)
        
        print("KSC side-by-side comparison saved as: gis-maps/ksc_chicago_pm25_comparison.html")
        return mae, rmse, r2
    
    def calculate_and_display_metrics(self, test_results):
        """
        Calculate and display performance metrics
        """
        y_true = test_results['y_true']
        y_pred = test_results['y_pred']
        locations = test_results['locations']
        errors = test_results['errors']
        
        # Overall metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print("\n" + "="*70)
        print("KSC-CONVLSTM CHICAGO AIR QUALITY MAPPING RESULTS")
        print("="*70)
        print(f"Model Architecture: K-Nearest Neighbors Spatial Convolution + ConvLSTM")
        print(f"Test samples: {len(y_true)}")
        print(f"Mean Absolute Error (MAE): {mae:.3f} μg/m³")
        print(f"Root Mean Square Error (RMSE): {rmse:.3f} μg/m³")
        print(f"R² Score: {r2:.3f}")
        print(f"Mean PM2.5 (Actual): {np.mean(y_true):.2f} μg/m³")
        print(f"Mean PM2.5 (Predicted): {np.mean(y_pred):.2f} μg/m³")
        
        # Location-wise performance
        print(f"\nLocation-wise Performance Analysis:")
        print("-" * 50)
        
        # Get unique locations and their average errors
        location_data = {}
        for i, location in enumerate(locations):
            if location not in location_data:
                location_data[location] = {'errors': [], 'actual': [], 'predicted': []}
            location_data[location]['errors'].append(errors[i])
            location_data[location]['actual'].append(y_true[i])
            location_data[location]['predicted'].append(y_pred[i])
        
        location_summary = []
        for location, data in location_data.items():
            avg_error = np.mean(data['errors'])
            avg_actual = np.mean(data['actual'])
            avg_predicted = np.mean(data['predicted'])
            location_summary.append((location, avg_error, avg_actual, avg_predicted))
        
        # Sort by error
        location_summary.sort(key=lambda x: x[1])
        
        print("🎯 Best Predictions (lowest average error):")
        for i, (location, error, actual, pred) in enumerate(location_summary[:3]):
            print(f"  {i+1}. {location}")
            print(f"     Avg Actual: {actual:.2f} μg/m³, Avg Predicted: {pred:.2f} μg/m³, Avg Error: {error:.2f} μg/m³")
        
        print("\n⚠️  Challenging Predictions (highest average error):")
        for i, (location, error, actual, pred) in enumerate(location_summary[-3:]):
            print(f"  {i+1}. {location}")
            print(f"     Avg Actual: {actual:.2f} μg/m³, Avg Predicted: {pred:.2f} μg/m³, Avg Error: {error:.2f} μg/m³")
        
        return mae, rmse, r2

def main():
    """
    Main function to create KSC Chicago air quality prediction maps
    """
    print("Starting KSC Chicago Air Quality Mapping")
    print("="*60)
    
    # Initialize the mapper
    mapper = KSCChicagoMapper()
    
    # Load KSC predictions
    test_results = mapper.load_ksc_predictions()
    
    if test_results is None:
        print("Failed to load KSC predictions. Exiting.")
        return
    
    # Calculate and display metrics
    mae, rmse, r2 = mapper.calculate_and_display_metrics(test_results)
    
    # Create comparison maps
    map_actual, map_predicted = mapper.create_comparison_maps(test_results)
    
    # Create side-by-side HTML
    mapper.create_side_by_side_html(test_results)
    
    print("\n" + "="*70)
    print("KSC MAPPING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("Generated files:")
    print("1. gis-maps/ksc_chicago_actual_pm
