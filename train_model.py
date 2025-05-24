# Import the data preparation functions
from data_preparation import prepare_air_quality_data_for_training, train_model_on_real_data
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Load your data (adjust path as needed)
print("Loading air quality data...")
combined_df = pd.read_csv('grid_data/combined.csv', index_col=[0, 1, 2, 3])

# Define the advanced KSC-ConvLSTM model components
class KNNReducer(layers.Layer):
    """Properly handles batch and time dimensions for spatial filtering"""
    def __init__(self, knn_indices_1d, **kwargs):
        super().__init__(**kwargs)
        self.knn_indices = tf.constant(knn_indices_1d, dtype=tf.int32)
        
    def call(self, inputs):
        batch, time, h, w, features = tf.unstack(tf.shape(inputs))
        x = tf.reshape(inputs, [batch*time, h, w, features])
        flat_x = tf.reshape(x, [batch*time, h*w, features])
        neighbors = tf.gather(flat_x, self.knn_indices, axis=1)
        reduced = tf.reduce_mean(neighbors, axis=2)
        return tf.reshape(reduced, [batch, time, h, w, features])

class SpatioTemporalAttention(layers.Layer):
    def build(self, input_shape):
        self.time_attention = layers.Attention(use_scale=True)
        self.spatial_attention = layers.Attention(use_scale=True)

    def call(self, inputs):
        batch, time, h, w, features = tf.unstack(tf.shape(inputs))
        x = tf.reshape(inputs, [batch, time, h * w, features])
        x = tf.transpose(x, [0, 2, 1, 3])
        x_reshaped = tf.reshape(x, [-1, time, features])
        t_att = self.time_attention([x_reshaped, x_reshaped])
        t_att = tf.reshape(t_att, [batch, h * w, time, features])
        t_att = tf.transpose(t_att, [0, 2, 1, 3])
        t_att = tf.reshape(t_att, [batch, time, h, w, features])

        x = tf.reshape(t_att, [batch, time, h * w, features])
        x_reshaped = tf.reshape(x, [-1, h * w, features])
        s_att = self.spatial_attention([x_reshaped, x_reshaped])
        return tf.reshape(s_att, [batch, time, h, w, features])

def residual_block(x):
    shortcut = x
    x = layers.LayerNormalization()(x)
    x = layers.ConvLSTM2D(64, (3,3), padding='same', return_sequences=True)(x)
    return layers.Add()([shortcut, x])

def build_ksc_convlstm(knn_indices_1d, grid_h, grid_w, seq_length=12, features=5):
    inputs = layers.Input(shape=(seq_length, grid_h, grid_w, features))
    
    # 1. KNN Spatial Filtering
    x = KNNReducer(knn_indices_1d)(inputs)
    
    # 2. ConvLSTM backbone
    x = layers.ConvLSTM2D(
        64, (3,3), padding='same', 
        return_sequences=True, 
        activation='tanh'
    )(x)
    
    # 3. Residual blocks with attention
    for _ in range(2):
        x = residual_block(x)
    x = SpatioTemporalAttention()(x)
    
    # 4. Final prediction
    x = layers.ConvLSTM2D(64, (3,3), padding='same', return_sequences=False)(x)
    outputs = layers.Dense(1, activation='linear')(x)  # Changed to linear for regression
    
    return Model(inputs, outputs)

def create_knn_indices(grid_h, grid_w, k=5):
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # MAE
    axes[0, 1].plot(history.history['mae'], label='Training MAE')
    axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
    axes[0, 1].set_title('Model MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    
    # MSE
    axes[1, 0].plot(history.history['mse'], label='Training MSE')
    axes[1, 0].plot(history.history['val_mse'], label='Validation MSE')
    axes[1, 0].set_title('Model MSE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    GRID_H, GRID_W = 50, 50
    SEQ_LENGTH = 12
    FEATURES = 5
    K = 5
    EPOCHS = 30
    
    print("Building KSC-ConvLSTM model...")
    
    # Generate KNN indices
    knn_1d = create_knn_indices(GRID_H, GRID_W, k=K)
    
    # Build model
    model = build_ksc_convlstm(
        knn_indices_1d=knn_1d,
        grid_h=GRID_H,
        grid_w=GRID_W,
        seq_length=SEQ_LENGTH,
        features=FEATURES
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae', 'mse']
    )
    
    print("Model architecture:")
    model.summary()
    
    # Train model on real data
    print("\nTraining model on real air quality data...")
    history, scaler = train_model_on_real_data(
        combined_df, 
        model, 
        grid_h=GRID_H, 
        grid_w=GRID_W, 
        seq_length=SEQ_LENGTH, 
        epochs=EPOCHS
    )
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    # Save the trained model
    print("Saving trained model...")
    model.save('ksc_convlstm_air_quality_model.h5')
    
    # Save the scaler for future use
    import joblib
    joblib.dump(scaler, 'air_quality_scaler.pkl')
    
    print("Training completed successfully!")
    print("Model saved as: ksc_convlstm_air_quality_model.h5")
    print("Scaler saved as: air_quality_scaler.pkl")
    
    return model, history, scaler

if __name__ == "__main__":
    model, history, scaler = main()
