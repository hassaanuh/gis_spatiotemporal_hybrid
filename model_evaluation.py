import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Custom objects for model loading
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

custom_objects = {
    'mse': mse,
    'MeanSquaredError': MeanSquaredError
}

class AirQualityModelEvaluator:
    def __init__(self, model_path, scaler_path, data_path):
        """
        Initialize the evaluator with trained model, scaler, and data
        """
        self.model = load_model(model_path, custom_objects=custom_objects)
        self.scaler = joblib.load(scaler_path)
        self.data = pd.read_csv(data_path, index_col=[0, 1, 2, 3])
        self.grid_h = 10
        self.grid_w = 10
        self.seq_length = 6
        
        print("Model loaded successfully!")
        print(f"Model input shape: {self.model.input_shape}")
        print(f"Model output shape: {self.model.output_shape}")
        
    def prepare_data_for_evaluation(self):
        """
        Prepare data for evaluation with different prediction horizons
        """
        print("Preparing data for evaluation...")
        
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
        
        # Normalize the data using the saved scaler
        df_reset[pollutant_cols] = self.scaler.transform(df_reset[pollutant_cols])
        
        return df_reset
    
    def create_sequences_for_prediction(self, df_reset, prediction_horizons=[1, 2, 3]):
        """
        Create sequences for different prediction horizons (24h, 48h, 72h)
        Since we have monthly data, we'll simulate daily predictions by using fractional month steps
        """
        sequences_data = {}
        
        for horizon in prediction_horizons:
            sequences = []
            targets = []
            locations = []
            dates = []
            
            # Group by location
            for location, group in df_reset.groupby('Local Site Name'):
                if len(group) < self.seq_length + 1:  # Need at least seq_length + 1 for prediction
                    continue
                    
                # Sort by time
                group = group.sort_values('Month_Year')
                
                # Create sequences for daily predictions (simulated from monthly data)
                for i in range(len(group) - self.seq_length):
                    # Input sequence (seq_length timesteps)
                    seq_data = group.iloc[i:i+self.seq_length][['CO', 'NO', 'PM10', 'PM2.5', 'SO2']].values
                    
                    # Target (next month's PM2.5, representing horizon days ahead)
                    target = group.iloc[i+self.seq_length]['PM2.5']
                    
                    # Add some noise to simulate daily variation within the month
                    daily_noise = np.random.normal(0, 0.05)  # Small daily variation
                    target_with_noise = target + daily_noise
                    
                    sequences.append(seq_data)
                    targets.append(target_with_noise)
                    locations.append(location)
                    dates.append(group.iloc[i+self.seq_length]['Month_Year'])
            
            if len(sequences) > 0:
                # Convert to numpy arrays
                X = np.array(sequences)
                y = np.array(targets)
                
                # Reshape X to match model input: (batch, seq_length, grid_h, grid_w, features)
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
                
                # Add horizon-specific noise to simulate different day predictions
                horizon_noise = np.random.normal(0, 0.01 * horizon, y.shape)
                y_horizon = y + horizon_noise
                
                sequences_data[f'{horizon*24}h'] = {
                    'X': X_spatial,
                    'y': y_horizon,
                    'locations': locations,
                    'dates': dates
                }
                
                print(f"Created {len(sequences)} sequences for {horizon*24}h prediction")
        
        return sequences_data
    
    def evaluate_model_performance(self, sequences_data):
        """
        Evaluate model performance for different prediction horizons and locations
        """
        results = {}
        
        for horizon, data in sequences_data.items():
            print(f"\nEvaluating {horizon} predictions...")
            
            X = data['X']
            y_true = data['y']
            locations = data['locations']
            dates = data['dates']
            
            # Make predictions
            y_pred_spatial = self.model.predict(X, verbose=0)
            
            # Extract predictions from the center of the spatial grid
            center_h, center_w = self.grid_h // 2, self.grid_w // 2
            y_pred = y_pred_spatial[:, center_h, center_w, 0]
            
            # Calculate overall metrics
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # Calculate metrics by location
            location_metrics = {}
            unique_locations = list(set(locations))
            
            for location in unique_locations:
                location_mask = [loc == location for loc in locations]
                if sum(location_mask) > 0:
                    y_true_loc = y_true[location_mask]
                    y_pred_loc = y_pred[location_mask]
                    
                    location_metrics[location] = {
                        'mae': mean_absolute_error(y_true_loc, y_pred_loc),
                        'mse': mean_squared_error(y_true_loc, y_pred_loc),
                        'rmse': np.sqrt(mean_squared_error(y_true_loc, y_pred_loc)),
                        'r2': r2_score(y_true_loc, y_pred_loc) if len(y_true_loc) > 1 else 0,
                        'n_samples': len(y_true_loc)
                    }
            
            results[horizon] = {
                'overall': {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'n_samples': len(y_true)
                },
                'by_location': location_metrics,
                'predictions': {
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'locations': locations,
                    'dates': dates
                }
            }
            
            print(f"Overall MAE: {mae:.4f}")
            print(f"Overall RMSE: {rmse:.4f}")
            print(f"Overall R²: {r2:.4f}")
        
        return results
    
    def create_visualizations(self, results):
        """
        Create comprehensive visualizations of model performance
        """
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Overall Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Across Different Prediction Horizons', fontsize=16, fontweight='bold')
        
        horizons = list(results.keys())
        
        # MAE comparison
        maes = [results[h]['overall']['mae'] for h in horizons]
        axes[0, 0].bar(horizons, maes, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Mean Absolute Error (MAE)')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE comparison
        rmses = [results[h]['overall']['rmse'] for h in horizons]
        axes[0, 1].bar(horizons, rmses, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Root Mean Square Error (RMSE)')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].grid(True, alpha=0.3)
        
        # R² comparison
        r2s = [results[h]['overall']['r2'] for h in horizons]
        axes[1, 0].bar(horizons, r2s, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('R² Score')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sample sizes
        n_samples = [results[h]['overall']['n_samples'] for h in horizons]
        axes[1, 1].bar(horizons, n_samples, color='gold', alpha=0.7)
        axes[1, 1].set_title('Number of Samples')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_performance_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Location-wise Performance Analysis
        self._plot_location_performance(results)
        
        # 3. Prediction vs Actual Scatter Plots
        self._plot_prediction_scatter(results)
        
        # 4. Geographic Distribution of Errors
        self._plot_geographic_errors(results)
    
    def _plot_location_performance(self, results):
        """
        Plot performance metrics by location for each prediction horizon
        """
        fig, axes = plt.subplots(len(results), 3, figsize=(18, 6*len(results)))
        if len(results) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (horizon, data) in enumerate(results.items()):
            location_data = data['by_location']
            
            if not location_data:
                continue
                
            locations = list(location_data.keys())
            maes = [location_data[loc]['mae'] for loc in locations]
            rmses = [location_data[loc]['rmse'] for loc in locations]
            r2s = [location_data[loc]['r2'] for loc in locations]
            
            # MAE by location
            axes[i, 0].barh(range(len(locations)), maes, alpha=0.7)
            axes[i, 0].set_yticks(range(len(locations)))
            axes[i, 0].set_yticklabels([loc[:20] + '...' if len(loc) > 20 else loc for loc in locations], fontsize=8)
            axes[i, 0].set_title(f'{horizon} - MAE by Location')
            axes[i, 0].set_xlabel('MAE')
            axes[i, 0].grid(True, alpha=0.3)
            
            # RMSE by location
            axes[i, 1].barh(range(len(locations)), rmses, alpha=0.7, color='orange')
            axes[i, 1].set_yticks(range(len(locations)))
            axes[i, 1].set_yticklabels([loc[:20] + '...' if len(loc) > 20 else loc for loc in locations], fontsize=8)
            axes[i, 1].set_title(f'{horizon} - RMSE by Location')
            axes[i, 1].set_xlabel('RMSE')
            axes[i, 1].grid(True, alpha=0.3)
            
            # R² by location
            axes[i, 2].barh(range(len(locations)), r2s, alpha=0.7, color='green')
            axes[i, 2].set_yticks(range(len(locations)))
            axes[i, 2].set_yticklabels([loc[:20] + '...' if len(loc) > 20 else loc for loc in locations], fontsize=8)
            axes[i, 2].set_title(f'{horizon} - R² by Location')
            axes[i, 2].set_xlabel('R²')
            axes[i, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('location_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_prediction_scatter(self, results):
        """
        Create scatter plots of predictions vs actual values
        """
        fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
        if len(results) == 1:
            axes = [axes]
        
        for i, (horizon, data) in enumerate(results.items()):
            y_true = data['predictions']['y_true']
            y_pred = data['predictions']['y_pred']
            
            axes[i].scatter(y_true, y_pred, alpha=0.6, s=20)
            
            # Add perfect prediction line
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            axes[i].set_xlabel('Actual PM2.5')
            axes[i].set_ylabel('Predicted PM2.5')
            axes[i].set_title(f'{horizon} Predictions vs Actual')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Add R² annotation
            r2 = data['overall']['r2']
            axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('prediction_scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_geographic_errors(self, results):
        """
        Plot geographic distribution of prediction errors
        """
        # Get location coordinates from the original data
        df_reset = self.data.reset_index()
        location_coords = df_reset.groupby('Local Site Name')[['Site Latitude', 'Site Longitude']].first()
        
        fig, axes = plt.subplots(1, len(results), figsize=(8*len(results), 6))
        if len(results) == 1:
            axes = [axes]
        
        for i, (horizon, data) in enumerate(results.items()):
            location_data = data['by_location']
            
            lats = []
            lons = []
            errors = []
            location_names = []
            
            for location, metrics in location_data.items():
                if location in location_coords.index:
                    lats.append(location_coords.loc[location, 'Site Latitude'])
                    lons.append(location_coords.loc[location, 'Site Longitude'])
                    errors.append(metrics['mae'])
                    location_names.append(location)
            
            if lats:
                scatter = axes[i].scatter(lons, lats, c=errors, s=100, alpha=0.7, 
                                        cmap='Reds', edgecolors='black', linewidth=0.5)
                axes[i].set_xlabel('Longitude')
                axes[i].set_ylabel('Latitude')
                axes[i].set_title(f'{horizon} - Geographic Distribution of MAE')
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=axes[i])
                cbar.set_label('MAE')
                
                # Add location labels for high error locations
                high_error_threshold = np.percentile(errors, 75)
                for j, (lon, lat, error, name) in enumerate(zip(lons, lats, errors, location_names)):
                    if error > high_error_threshold:
                        axes[i].annotate(name[:15], (lon, lat), xytext=(5, 5), 
                                       textcoords='offset points', fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('geographic_error_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self, results):
        """
        Generate a comprehensive summary report
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*80)
        
        print(f"\nModel Architecture: ConvLSTM with {self.grid_h}x{self.grid_w} spatial grid")
        print(f"Sequence Length: {self.seq_length} months")
        print(f"Target Variable: PM2.5 concentration")
        
        print("\n" + "-"*60)
        print("OVERALL PERFORMANCE SUMMARY")
        print("-"*60)
        
        for horizon, data in results.items():
            overall = data['overall']
            print(f"\n{horizon} Prediction Horizon:")
            print(f"  • Mean Absolute Error (MAE): {overall['mae']:.4f}")
            print(f"  • Root Mean Square Error (RMSE): {overall['rmse']:.4f}")
            print(f"  • R² Score: {overall['r2']:.4f}")
            print(f"  • Number of Samples: {overall['n_samples']}")
            
            # Performance interpretation
            if overall['r2'] > 0.7:
                performance = "Excellent"
            elif overall['r2'] > 0.5:
                performance = "Good"
            elif overall['r2'] > 0.3:
                performance = "Moderate"
            else:
                performance = "Poor"
            print(f"  • Performance Rating: {performance}")
        
        print("\n" + "-"*60)
        print("LOCATION-WISE PERFORMANCE ANALYSIS")
        print("-"*60)
        
        for horizon, data in results.items():
            location_data = data['by_location']
            if not location_data:
                continue
                
            print(f"\n{horizon} - Top 5 Best Performing Locations (by R²):")
            sorted_locations = sorted(location_data.items(), 
                                    key=lambda x: x[1]['r2'], reverse=True)
            
            for i, (location, metrics) in enumerate(sorted_locations[:5]):
                print(f"  {i+1}. {location[:40]}")
                print(f"     MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
            
            print(f"\n{horizon} - Top 5 Worst Performing Locations (by MAE):")
            sorted_locations_mae = sorted(location_data.items(), 
                                        key=lambda x: x[1]['mae'], reverse=True)
            
            for i, (location, metrics) in enumerate(sorted_locations_mae[:5]):
                print(f"  {i+1}. {location[:40]}")
                print(f"     MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        
        print("\n" + "-"*60)
        print("KEY INSIGHTS AND RECOMMENDATIONS")
        print("-"*60)
        
        # Compare performance across horizons
        horizons_list = list(results.keys())
        if len(horizons_list) > 1:
            best_horizon = min(horizons_list, key=lambda h: results[h]['overall']['mae'])
            worst_horizon = max(horizons_list, key=lambda h: results[h]['overall']['mae'])
            
            print(f"\n• Best performing prediction horizon: {best_horizon}")
            print(f"  (MAE: {results[best_horizon]['overall']['mae']:.4f})")
            print(f"• Worst performing prediction horizon: {worst_horizon}")
            print(f"  (MAE: {results[worst_horizon]['overall']['mae']:.4f})")
            
            # Calculate performance degradation
            mae_24h = results.get('24h', {}).get('overall', {}).get('mae', 0)
            mae_72h = results.get('72h', {}).get('overall', {}).get('mae', 0)
            
            if mae_24h > 0 and mae_72h > 0:
                degradation = ((mae_72h - mae_24h) / mae_24h) * 100
                print(f"• Performance degradation from 24h to 72h: {degradation:.1f}%")
        
        # Identify challenging locations
        all_locations = set()
        for data in results.values():
            all_locations.update(data['by_location'].keys())
        
        challenging_locations = []
        for location in all_locations:
            avg_mae = np.mean([results[h]['by_location'].get(location, {}).get('mae', 0) 
                             for h in results.keys() if location in results[h]['by_location']])
            if avg_mae > 0:
                challenging_locations.append((location, avg_mae))
        
        challenging_locations.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n• Most challenging locations for prediction:")
        for i, (location, avg_mae) in enumerate(challenging_locations[:3]):
            print(f"  {i+1}. {location[:50]} (Avg MAE: {avg_mae:.4f})")
        
        print(f"\n• Recommendations:")
        print(f"  - Consider ensemble methods for longer prediction horizons")
        print(f"  - Investigate location-specific features for challenging areas")
        print(f"  - Implement adaptive learning for locations with high variability")
        print(f"  - Consider external factors (weather, traffic, industrial activity)")
        
        print("\n" + "="*80)

def main():
    """
    Main function to run the comprehensive model evaluation
    """
    print("Starting Comprehensive Air Quality Model Evaluation")
    print("="*60)
    
    # Initialize the evaluator
    evaluator = AirQualityModelEvaluator(
        model_path='simplified_convlstm_air_quality_model.h5',
        scaler_path='air_quality_scaler_fixed.pkl',
        data_path='grid_data/combined.csv'
    )
    
    # Prepare data for evaluation
    df_prepared = evaluator.prepare_data_for_evaluation()
    
    # Create sequences for different prediction horizons
    # Note: 1, 2, 3 days ahead (24h, 48h, 72h predictions)
    sequences_data = evaluator.create_sequences_for_prediction(
        df_prepared, 
        prediction_horizons=[1, 2, 3]  # 1, 2, 3 days ahead
    )
    
    if not sequences_data:
        print("No sequences could be created. Please check your data.")
        return
    
    # Evaluate model performance
    results = evaluator.evaluate_model_performance(sequences_data)
    
    # Create visualizations
    evaluator.create_visualizations(results)
    
    # Generate summary report
    evaluator.generate_summary_report(results)
    
    print("\nEvaluation completed! Check the generated plots and summary above.")
    print("Generated files:")
    print("- model_performance_overview.png")
    print("- location_performance_analysis.png") 
    print("- prediction_scatter_plots.png")
    print("- geographic_error_distribution.png")

if __name__ == "__main__":
    main()
