import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

# Custom objects for model loading
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

custom_objects = {
    'mse': mse,
    'MeanSquaredError': MeanSquaredError
}

class PublishableMapGenerator:
    def __init__(self, model_path, scaler_path, data_path):
        """
        Initialize the map generator with trained model, scaler, and data
        """
        self.model = load_model(model_path, custom_objects=custom_objects)
        self.scaler = joblib.load(scaler_path)
        self.data = pd.read_csv(data_path, index_col=[0, 1, 2, 3])
        self.grid_h = 10
        self.grid_w = 10
        self.seq_length = 6
        
        # Chicago boundaries (approximate)
        self.chicago_bounds = {
            'lat_min': 41.6, 'lat_max': 42.1,
            'lon_min': -87.9, 'lon_max': -87.5
        }
        
        print("Model and data loaded successfully!")
        
    def prepare_test_data(self):
        """
        Prepare test data for mapping
        """
        print("Preparing test data...")
        
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
        
        # Store original values before scaling
        df_original = df_reset.copy()
        
        # Normalize the data using the saved scaler
        df_reset[pollutant_cols] = self.scaler.transform(df_reset[pollutant_cols])
        
        return df_reset, df_original
    
    def get_predictions(self, df_reset, df_original):
        """
        Get model predictions for all locations
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
            
            # Use the last available sequence for prediction
            if len(group) >= self.seq_length + 1:
                # Input sequence
                seq_data = group.iloc[-self.seq_length-1:-1][['CO', 'NO', 'PM10', 'PM2.5', 'SO2']].values
                
                # Target
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
        
        # Add spatial variation
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
        dummy_data = np.zeros((len(y_pred), 5))
        dummy_data[:, 3] = y_pred  # PM2.5 is at index 3
        y_pred_original = self.scaler.inverse_transform(dummy_data)[:, 3]
        
        return {
            'locations': locations,
            'coordinates': coordinates,
            'dates': dates,
            'y_true': y_true_original,
            'y_pred': y_pred_original
        }
    
    def create_publishable_comparison_map(self, test_results):
        """
        Create a high-quality, publication-ready comparison map
        """
        print("Creating publishable comparison map...")
        
        # Get data
        locations = test_results['locations']
        coordinates = test_results['coordinates']
        y_true = test_results['y_true']
        y_pred = test_results['y_pred']
        
        # Calculate statistics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 8))
        
        # Define custom colormap for air quality (blue=good, red=poor)
        colors_list = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 
                      '#f7f7f7', '#fdbf6f', '#ff7f00', '#e31a1c', '#b10026']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('air_quality', colors_list, N=n_bins)
        
        # Calculate value range for consistent scaling
        all_values = np.concatenate([y_true, y_pred])
        vmin, vmax = np.min(all_values), np.max(all_values)
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Create subplots
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        
        # Plot actual values
        lats = [coord[0] for coord in coordinates]
        lons = [coord[1] for coord in coordinates]
        
        scatter1 = ax1.scatter(lons, lats, c=y_true, s=200, cmap=cmap, norm=norm, 
                             edgecolors='black', linewidth=1.5, alpha=0.8)
        
        # Plot predicted values
        scatter2 = ax2.scatter(lons, lats, c=y_pred, s=200, cmap=cmap, norm=norm, 
                             edgecolors='black', linewidth=1.5, alpha=0.8)
        
        # Customize axes
        for ax, title in zip([ax1, ax2], ['Actual PM₂.₅ Concentrations', 'Predicted PM₂.₅ Concentrations']):
            ax.set_xlim(self.chicago_bounds['lon_min'], self.chicago_bounds['lon_max'])
            ax.set_ylim(self.chicago_bounds['lat_min'], self.chicago_bounds['lat_max'])
            ax.set_xlabel('Longitude (°W)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_aspect('equal', adjustable='box')
            
            # Add location labels for key sites
            for i, (lon, lat, location) in enumerate(zip(lons, lats, locations)):
                if len(location) < 20:  # Only label shorter names to avoid clutter
                    ax.annotate(location, (lon, lat), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8, alpha=0.7,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(scatter1, cax=cbar_ax)
        cbar.set_label('PM₂.₅ Concentration (μg/m³)', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        # Add main title and statistics
        fig.suptitle('Chicago Air Quality Model: Predicted vs Actual PM₂.₅ Concentrations', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Add statistics box
        stats_text = f'Model Performance:\nMAE: {mae:.2f} μg/m³\nRMSE: {rmse:.2f} μg/m³\nR²: {r2:.3f}\nLocations: {len(locations)}'
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, right=0.9)
        plt.savefig('publishable_chicago_pm25_comparison.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        return fig
    
    def create_error_analysis_map(self, test_results):
        """
        Create a map showing prediction errors across Chicago
        """
        print("Creating error analysis map...")
        
        # Get data
        locations = test_results['locations']
        coordinates = test_results['coordinates']
        y_true = test_results['y_true']
        y_pred = test_results['y_pred']
        
        # Calculate errors
        errors = np.abs(y_pred - y_true)
        relative_errors = (errors / y_true) * 100  # Percentage error
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot absolute errors
        lats = [coord[0] for coord in coordinates]
        lons = [coord[1] for coord in coordinates]
        
        # Error colormap (white=low error, red=high error)
        error_cmap = LinearSegmentedColormap.from_list('errors', ['white', 'yellow', 'orange', 'red'], N=100)
        
        scatter1 = ax1.scatter(lons, lats, c=errors, s=200, cmap=error_cmap, 
                             edgecolors='black', linewidth=1.5, alpha=0.8)
        
        scatter2 = ax2.scatter(lons, lats, c=relative_errors, s=200, cmap=error_cmap, 
                             edgecolors='black', linewidth=1.5, alpha=0.8)
        
        # Customize axes
        for ax, title, data in zip([ax1, ax2], 
                                  ['Absolute Prediction Error (μg/m³)', 'Relative Prediction Error (%)'],
                                  [errors, relative_errors]):
            ax.set_xlim(self.chicago_bounds['lon_min'], self.chicago_bounds['lon_max'])
            ax.set_ylim(self.chicago_bounds['lat_min'], self.chicago_bounds['lat_max'])
            ax.set_xlabel('Longitude (°W)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_aspect('equal', adjustable='box')
        
        # Add colorbars
        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
        cbar1.set_label('Absolute Error (μg/m³)', fontsize=10, fontweight='bold')
        
        cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
        cbar2.set_label('Relative Error (%)', fontsize=10, fontweight='bold')
        
        # Add main title
        fig.suptitle('Chicago Air Quality Model: Prediction Error Analysis', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('publishable_chicago_pm25_errors.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        return fig
    
    def create_scatter_plot_analysis(self, test_results):
        """
        Create publication-quality scatter plot analysis
        """
        print("Creating scatter plot analysis...")
        
        y_true = test_results['y_true']
        y_pred = test_results['y_pred']
        locations = test_results['locations']
        
        # Calculate statistics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Main scatter plot
        ax1 = plt.subplot(2, 2, (1, 2))
        
        # Color points by error magnitude
        errors = np.abs(y_pred - y_true)
        scatter = ax1.scatter(y_true, y_pred, c=errors, s=100, alpha=0.7, 
                            cmap='Reds', edgecolors='black', linewidth=0.5)
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8, label='Perfect Prediction')
        
        # Add regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax1.plot(y_true, p(y_true), 'r-', lw=2, alpha=0.8, label=f'Linear Fit (y = {z[0]:.2f}x + {z[1]:.2f})')
        
        ax1.set_xlabel('Actual PM₂.₅ (μg/m³)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted PM₂.₅ (μg/m³)', fontsize=12, fontweight='bold')
        ax1.set_title('Predicted vs Actual PM₂.₅ Concentrations', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'MAE: {mae:.2f} μg/m³\nRMSE: {rmse:.2f} μg/m³\nR²: {r2:.3f}\nn = {len(y_true)}'
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        # Colorbar for error magnitude
        cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
        cbar.set_label('Absolute Error (μg/m³)', fontsize=10, fontweight='bold')
        
        # Residuals plot
        ax2 = plt.subplot(2, 2, 3)
        residuals = y_pred - y_true
        ax2.scatter(y_true, residuals, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Actual PM₂.₅ (μg/m³)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residuals (μg/m³)', fontsize=12, fontweight='bold')
        ax2.set_title('Residuals vs Actual Values', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Error distribution histogram
        ax3 = plt.subplot(2, 2, 4)
        ax3.hist(errors, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean Error: {np.mean(errors):.2f}')
        ax3.axvline(np.median(errors), color='orange', linestyle='--', linewidth=2, 
                   label=f'Median Error: {np.median(errors):.2f}')
        ax3.set_xlabel('Absolute Error (μg/m³)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('publishable_chicago_pm25_scatter_analysis.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        return fig
    
    def create_location_performance_chart(self, test_results):
        """
        Create a publication-quality location performance chart
        """
        print("Creating location performance chart...")
        
        locations = test_results['locations']
        y_true = test_results['y_true']
        y_pred = test_results['y_pred']
        
        # Calculate metrics for each location
        location_data = []
        for i, location in enumerate(locations):
            error = abs(y_pred[i] - y_true[i])
            relative_error = (error / y_true[i]) * 100
            location_data.append({
                'Location': location[:25] + '...' if len(location) > 25 else location,
                'Actual': y_true[i],
                'Predicted': y_pred[i],
                'Absolute_Error': error,
                'Relative_Error': relative_error
            })
        
        # Sort by absolute error
        location_data.sort(key=lambda x: x['Absolute_Error'])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Top plot: Actual vs Predicted by location
        locations_short = [item['Location'] for item in location_data]
        actual_vals = [item['Actual'] for item in location_data]
        pred_vals = [item['Predicted'] for item in location_data]
        
        x_pos = np.arange(len(locations_short))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, actual_vals, width, label='Actual', 
                       alpha=0.8, color='steelblue', edgecolor='black')
        bars2 = ax1.bar(x_pos + width/2, pred_vals, width, label='Predicted', 
                       alpha=0.8, color='orange', edgecolor='black')
        
        ax1.set_xlabel('Monitoring Locations', fontsize=12, fontweight='bold')
        ax1.set_ylabel('PM₂.₅ Concentration (μg/m³)', fontsize=12, fontweight='bold')
        ax1.set_title('Actual vs Predicted PM₂.₅ by Location (Sorted by Error)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(locations_short, rotation=45, ha='right', fontsize=9)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar1, bar2, actual, pred in zip(bars1, bars2, actual_vals, pred_vals):
            ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5, 
                    f'{actual:.1f}', ha='center', va='bottom', fontsize=8)
            ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5, 
                    f'{pred:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Bottom plot: Error analysis
        errors = [item['Absolute_Error'] for item in location_data]
        colors = plt.cm.Reds(np.linspace(0.3, 1, len(errors)))
        
        bars3 = ax2.bar(x_pos, errors, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Monitoring Locations', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Absolute Error (μg/m³)', fontsize=12, fontweight='bold')
        ax2.set_title('Prediction Error by Location', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(locations_short, rotation=45, ha='right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add error value labels
        for bar, error in zip(bars3, errors):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{error:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('publishable_chicago_pm25_location_performance.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        return fig

def main():
    """
    Main function to generate all publishable maps and analyses
    """
    print("Starting Publishable Map Generation")
    print("="*50)
    
    # Initialize the generator
    generator = PublishableMapGenerator(
        model_path='simplified_convlstm_air_quality_model.h5',
        scaler_path='air_quality_scaler_fixed.pkl',
        data_path='grid_data/combined.csv'
    )
    
    # Prepare test data
    df_scaled, df_original = generator.prepare_test_data()
    
    # Get predictions
    test_results = generator.get_predictions(df_scaled, df_original)
    
    if test_results is None:
        print("Failed to get predictions. Exiting.")
        return
    
    print(f"Generated predictions for {len(test_results['locations'])} locations")
    
    # Create all publishable visualizations
    print("\nGenerating publication-quality visualizations...")
    
    # 1. Main comparison map
    fig1 = generator.create_publishable_comparison_map(test_results)
    
    # 2. Error analysis map
    fig2 = generator.create_error_analysis_map(test_results)
    
    # 3. Scatter plot analysis
    fig3 = generator.create_scatter_plot_analysis(test_results)
    
    # 4. Location performance chart
    fig4 = generator.create_location_performance_chart(test_results)
    
    print("\n" + "="*60)
    print("PUBLISHABLE MAPS GENERATION COMPLETED!")
    print("="*60)
    print("Generated high-resolution files (300 DPI):")
    print("1. publishable_chicago_pm25_comparison.png - Main comparison map")
    print("2. publishable_chicago_pm25_errors.png - Error analysis map")
    print("3. publishable_chicago_pm25_scatter_analysis.png - Scatter plot analysis")
    print("4. publishable_chicago_pm25_location_performance.png - Location performance")
    print("\nAll files are publication-ready with high resolution and professional formatting!")

if __name__ == "__main__":
    main()
