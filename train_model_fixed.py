# Import the data preparation functions
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load your data (adjust path as needed)
print("Loading air quality data...")
combined_df = pd.read_csv('grid_data/combined.csv', index_col=[0, 1, 2, 3])

def prepare_air_quality_data_fixed(combined_df, grid_h=10, grid_w=10, seq_length=6):
    """
    Prepare air quality data for spatiotemporal model training with better missing data handling
    """
    print("Preparing data with improved missing value handling...")
    
    # Reset index to work with the data
    df_reset = combined_df.reset_index()
    
    # Convert Month_Year to datetime
    df_reset['Month_Year'] = pd.to_datetime(df_reset['Month_Year'])
    
    # Sort by location and time
    df_reset = df_reset.sort_values(['Local Site Name', 'Month_Year'])
    
    # Define pollutant columns
    pollutant_cols = ['CO', 'NO', 'PM10', 'PM2.5', 'SO2']
    
    # Fill missing values more aggressively
    print("Handling missing values...")
    for col in pollutant_cols:
        # First fill with forward fill, then backward fill, then median
        df_reset[col] = df_reset.groupby('Local Site Name')[col].fillna(method='ffill')
        df_reset[col] = df_reset.groupby('Local Site Name')[col].fillna(method='bfill')
        # Fill remaining NaNs with overall median
        df_reset[col] = df_reset[col].fillna(df_reset[col].median())
    
    # Remove any remaining rows with NaN values
    df_reset = df_reset.dropna(subset=pollutant_cols)
    
    if len(df_reset) == 0:
        raise ValueError("No valid data remaining after cleaning!")
    
    print(f"Data after cleaning: {len(df_reset)} rows")
    
    # Normalize the data
    scaler = StandardScaler()
    df_reset[pollutant_cols] = scaler.fit_transform(df_reset[pollutant_cols])
    
    # Create sequences for each location
    sequences = []
    targets = []
    
    # Group by location
    location_count = 0
    for location, group in df_reset.groupby('Local Site Name'):
        if len(group) < seq_length + 1:  # Need at least seq_length + 1 for input + target
            continue
            
        location_count += 1
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
    
    print(f"Created sequences from {location_count} locations")
    
    if len(sequences) == 0:
        raise ValueError("No sequences could be created!")
    
    # Convert to numpy arrays
    X = np.array(sequences)  # Shape: (n_samples, seq_length, n_features)
    y = np.array(targets)    # Shape: (n_samples,)
    
    print(f"Sequence shape: {X.shape}, Target shape: {y.shape}")
    
    # Reshape X to match model input: (batch, seq_length, grid_h, grid_w, features)
    n_samples, seq_len, n_features = X.shape
    X_spatial = np.zeros((n_samples, seq_len, grid_h, grid_w, n_features))
    
    # Fill the center of each grid with the actual data
    center_h, center_w = grid_h // 2, grid_w // 2
    X_spatial[:, :, center_h, center_w, :] = X
    
    # Add some spatial variation by copying to nearby cells with small noise
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            h_idx = center_h + i
            w_idx = center_w + j
            if 0 <= h_idx < grid_h and 0 <= w_idx < grid_w:
                noise = np.random.normal(0, 0.05, X.shape)  # Reduced noise
                X_spatial[:, :, h_idx, w_idx, :] = X + noise
    
    # Reshape y to match model output: (batch, grid_h, grid_w, 1)
    y_spatial = np.zeros((n_samples, grid_h, grid_w, 1))
    y_spatial[:, center_h, center_w, 0] = y
    
    # Check for any remaining NaN or inf values
    if np.any(np.isnan(X_spatial)) or np.any(np.isinf(X_spatial)):
        print("Warning: NaN or inf values found in X_spatial")
        X_spatial = np.nan_to_num(X_spatial, nan=0.0, posinf=1.0, neginf=-1.0)
    
    if np.any(np.isnan(y_spatial)) or np.any(np.isinf(y_spatial)):
        print("Warning: NaN or inf values found in y_spatial")
        y_spatial = np.nan_to_num(y_spatial, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_spatial, y_spatial, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler

# Define the simplified KSC-ConvLSTM model components
class KNNReducer(layers.Layer):
    """Simplified KNN spatial filtering"""
    def __init__(self, knn_indices_1d, **kwargs):
        super().__init__(**kwargs)
        self.knn_indices = tf.constant(knn_indices_1d, dtype=tf.int32)
        
    def call(self, inputs):
        # For simplicity, just return the inputs without KNN processing for now
        return inputs

class SpatioTemporalAttention(layers.Layer):
    """Simplified attention mechanism"""
    def build(self, input_shape):
        super().build(input_shape)
        
    def call(self, inputs):
        # For simplicity, just return the inputs
        return inputs

def build_simplified_convlstm(grid_h, grid_w, seq_length=6, features=5):
    """Build a simplified ConvLSTM model"""
    inputs = layers.Input(shape=(seq_length, grid_h, grid_w, features))
    
    # ConvLSTM layers with proper regularization
    x = layers.ConvLSTM2D(
        32, (3,3), padding='same', 
        return_sequences=True, 
        activation='tanh',
        dropout=0.2,
        recurrent_dropout=0.2
    )(inputs)
    
    x = layers.BatchNormalization()(x)
    
    x = layers.ConvLSTM2D(
        32, (3,3), padding='same', 
        return_sequences=False,
        activation='tanh',
        dropout=0.2,
        recurrent_dropout=0.2
    )(x)
    
    x = layers.BatchNormalization()(x)
    
    # Final prediction layer
    outputs = layers.Dense(1, activation='linear')(x)
    
    return Model(inputs, outputs)

def create_knn_indices(grid_h, grid_w, k=3):
    """Generates 1D spatial indices for KNN filtering"""
    x = np.arange(grid_h)
    y = np.arange(grid_w)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    return indices[:, 1:]  # Exclude self

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # MAE
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_title('Model MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    
    # MSE
    axes[2].plot(history.history['mse'], label='Training MSE')
    axes[2].plot(history.history['val_mse'], label='Validation MSE')
    axes[2].set_title('Model MSE')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MSE')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('training_history_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Configuration - using smaller dimensions for stability
    GRID_H, GRID_W = 10, 10
    SEQ_LENGTH = 6
    FEATURES = 5
    EPOCHS = 20
    
    print("Building simplified ConvLSTM model...")
    
    try:
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = prepare_air_quality_data_fixed(
            combined_df, GRID_H, GRID_W, SEQ_LENGTH
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training target shape: {y_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test target shape: {y_test.shape}")
        
        # Build model
        model = build_simplified_convlstm(
            grid_h=GRID_H,
            grid_w=GRID_W,
            seq_length=SEQ_LENGTH,
            features=FEATURES
        )
        
        # Compile model with appropriate loss and metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',  # Using MSE instead of Huber for simplicity
            metrics=['mae', 'mse']
        )
        
        print("Model architecture:")
        model.summary()
        
        # Add callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        # Train model
        print("\nTraining model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=8,  # Smaller batch size
            verbose=1,
            callbacks=callbacks
        )
        
        # Evaluate the model
        print("Evaluating model...")
        test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        
        # Plot training history
        print("Plotting training history...")
        plot_training_history(history)
        
        # Save the trained model
        print("Saving trained model...")
        model.save('simplified_convlstm_air_quality_model.h5')
        
        # Save the scaler for future use
        import joblib
        joblib.dump(scaler, 'air_quality_scaler_fixed.pkl')
        
        print("Training completed successfully!")
        print("Model saved as: simplified_convlstm_air_quality_model.h5")
        print("Scaler saved as: air_quality_scaler_fixed.pkl")
        
        return model, history, scaler
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    model, history, scaler = main()
