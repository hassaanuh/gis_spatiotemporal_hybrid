import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.neighbors import NearestNeighbors

print('=== Running KSC-ConvLSTM Model ===')

# Load and prepare data
files = {
    'PM2.5': 'monthly/merged_2.5_monthly.csv',
    'CO': 'monthly/merged_co_monthly.csv',
    'SO2': 'monthly/merged_so2_monthly.csv',
    'NO': 'monthly/merged_no_monthly.csv',
    'PM10': 'monthly/merged_pm10_monthly.csv'
}

dfs = [pd.read_csv(f).assign(pollutant=name) for name, f in files.items()]
combined_df = pd.concat(dfs).pivot_table(index=['Local Site Name', 'Month_Year', 'Site Latitude', 'Site Longitude'],
                                        columns='pollutant', values='monthly_value')

print('Data loaded. Shape:', combined_df.shape)

# Reset index and clean data
df_reset = combined_df.reset_index()
df_clean = df_reset.dropna(subset=['PM2.5', 'Site Latitude', 'Site Longitude'])
print(f'Valid data points: {len(df_clean)}')

# Define KNN components
class KNNReducer(layers.Layer):
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

def create_knn_indices(grid_h, grid_w, k=5):
    x = np.arange(grid_h)
    y = np.arange(grid_w)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    return indices[:, 1:]

def build_ksc_convlstm(knn_indices_1d, grid_h, grid_w, seq_length=12, features=5):
    inputs = layers.Input(shape=(seq_length, grid_h, grid_w, features))
    
    # KNN Spatial Filtering
    x = KNNReducer(knn_indices_1d)(inputs)
    
    # ConvLSTM backbone
    x = layers.ConvLSTM2D(64, (3,3), padding='same', return_sequences=True, activation='tanh')(x)
    
    # Residual blocks with attention
    for _ in range(2):
        x = residual_block(x)
    x = SpatioTemporalAttention()(x)
    
    # Final prediction
    x = layers.ConvLSTM2D(64, (3,3), padding='same', return_sequences=False)(x)
    outputs = layers.Dense(1, activation='relu')(x)
    
    return Model(inputs, outputs)

# Configuration
GRID_H, GRID_W = 50, 50
SEQ_LENGTH = 12
FEATURES = 5
K = 5

print('Creating KNN indices...')
knn_1d = create_knn_indices(GRID_H, GRID_W, k=K)

print('Building KSC-ConvLSTM model...')
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

print('Model compiled successfully!')
print('Model summary:')
model.summary()

# Test with dummy data
print('\nTesting model with dummy data...')
dummy_input = tf.random.normal((32, 12, 50, 50, 5))
dummy_output = model(dummy_input)
print(f'Model output shape: {dummy_output.shape}')
print('Model test successful!')

print('\n=== KSC-ConvLSTM Model Run Complete ===')
