import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

def prepare_air_quality_data_for_training(combined_df, grid_h=50, grid_w=50, seq_length=12):
    """
    Prepare air quality data for spatiotemporal model training
    
    Args:
        combined_df: DataFrame with air quality data
        grid_h, grid_w: Grid dimensions
        seq_length: Sequence length for temporal modeling
    
    Returns:
        X_train, X_test, y_train, y_test: Training and testing data
    """
    
    # Reset index to work with the data
    df_reset = combined_df.reset_index()
    
    # Convert Month_Year to datetime
    df_reset['Month_Year'] = pd.to_datetime(df_reset['Month_Year'])
    
    # Sort by location and time
    df_reset = df_reset.sort_values(['Local Site Name', 'Month_Year'])
    
    # Fill missing values with forward fill then backward fill
    pollutant_cols = ['CO', 'NO', 'PM10', 'PM2.5', 'SO2']
    df_reset[pollutant_cols] = df_reset.groupby('Local Site Name')[pollutant_cols].fillna(method='ffill').fillna(method='bfill')
    
    # Normalize the data
    scaler = StandardScaler()
    df_reset[pollutant_cols] = scaler.fit_transform(df_reset[pollutant_cols])
    
    # Create sequences for each location
    sequences = []
    targets = []
    
    # Group by location
    for location, group in df_reset.groupby('Local Site Name'):
        if len(group) < seq_length + 1:  # Need at least seq_length + 1 for input + target
            continue
            
        # Sort by time
        group = group.sort_values('Month_Year')
        
        # Create sequences
        for i in range(len(group) - seq_length):
            # Input sequence (seq_length timesteps)
            seq_data = group.iloc[i:i+seq_length][pollutant_cols].values
            
            # Target (next timestep PM2.5)
            target = group.iloc[i+seq_length]['PM2.5']
            
            sequences.append(seq_data)
            targets.append(target)
    
    # Convert to numpy arrays
    X = np.array(sequences)  # Shape: (n_samples, seq_length, n_features)
    y = np.array(targets)    # Shape: (n_samples,)
    
    # Reshape X to match model input: (batch, seq_length, grid_h, grid_w, features)
    # For now, we'll replicate the temporal data across the spatial grid
    # In a real implementation, you'd interpolate to create proper spatial grids
    
    n_samples, seq_len, n_features = X.shape
    X_spatial = np.zeros((n_samples, seq_len, grid_h, grid_w, n_features))
    
    # Fill the center of each grid with the actual data
    center_h, center_w = grid_h // 2, grid_w // 2
    X_spatial[:, :, center_h, center_w, :] = X
    
    # Add some spatial variation by copying to nearby cells with small noise
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i == 0 and j == 0:
                continue
            h_idx = center_h + i
            w_idx = center_w + j
            if 0 <= h_idx < grid_h and 0 <= w_idx < grid_w:
                noise = np.random.normal(0, 0.1, X.shape)
                X_spatial[:, :, h_idx, w_idx, :] = X + noise
    
    # Reshape y to match model output: (batch, grid_h, grid_w, 1)
    y_spatial = np.zeros((n_samples, grid_h, grid_w, 1))
    y_spatial[:, center_h, center_w, 0] = y
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_spatial, y_spatial, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler

def create_temporal_sequences_from_grid_data(df, seq_length=12):
    """
    Create temporal sequences from the grid data for training
    """
    # Convert Month_Year to datetime and sort
    df['Month_Year'] = pd.to_datetime(df['Month_Year'])
    df = df.sort_values(['grid_id', 'Month_Year'])
    
    # Get unique grid IDs and time steps
    grid_ids = df['grid_id'].unique()
    time_steps = sorted(df['Month_Year'].unique())
    
    # Pollutant columns
    pollutant_cols = ['2.5', 'CO', 'NO', 'PM10', 'SO2']
    
    sequences = []
    targets = []
    
    for grid_id in grid_ids:
        grid_data = df[df['grid_id'] == grid_id].sort_values('Month_Year')
        
        if len(grid_data) < seq_length + 1:
            continue
            
        # Fill missing values
        grid_data[pollutant_cols] = grid_data[pollutant_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Create sequences
        for i in range(len(grid_data) - seq_length):
            seq_data = grid_data.iloc[i:i+seq_length][pollutant_cols].values
            target = grid_data.iloc[i+seq_length]['2.5']  # Predict PM2.5
            
            sequences.append(seq_data)
            targets.append(target)
    
    return np.array(sequences), np.array(targets)

# Example usage function
def train_model_on_real_data(combined_df, model, grid_h=50, grid_w=50, seq_length=12, epochs=50):
    """
    Train the model on real air quality data
    """
    print("Preparing training data...")
    X_train, X_test, y_train, y_test, scaler = prepare_air_quality_data_for_training(
        combined_df, grid_h, grid_w, seq_length
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test target shape: {y_test.shape}")
    
    # Train the model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=16,
        verbose=1
    )
    
    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    
    return history, scaler

if __name__ == "__main__":
    print("Data preparation utilities for air quality spatiotemporal modeling")
