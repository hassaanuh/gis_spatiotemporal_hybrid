import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print('=== Training KSC-ConvLSTM Model on Air Quality Data ===')

# Load and prepare data
files = {
    'PM2.5': 'monthly/merged_2.5_monthly.csv',
    'CO': 'monthly/merged_co_monthly.csv',
    'SO2': 'monthly/merged_so2_monthly.csv',
    'NO': 'monthly/merged_no_monthly.csv',
    'PM10': 'monthly/merged_pm10_monthly.csv'
}

print('Loading data files...')
dfs = []
for name, file in files.items():
    try:
        df = pd.read_csv(file)
        df['pollutant'] = name
        dfs.append(df)
        print(f'Loaded {name}: {len(df)} records')
    except Exception as e:
        print(f'Error loading {file}: {e}')

if not dfs:
    print('No data files could be loaded!')
    exit(1)

# Combine and pivot data
combined_df = pd.concat(dfs)
pivot_df = combined_df.pivot_table(
    index=['Local Site Name', 'Month_Year', 'Site Latitude', 'Site Longitude'],
    columns='pollutant', 
    values='monthly_value'
)

print(f'Combined data shape: {pivot_df.shape}')

# Reset index and clean data
df_reset = pivot_df.reset_index()
df_clean = df_reset.dropna(subset=['PM2.5', 'Site Latitude', 'Site Longitude'])
print(f'Valid data points after cleaning: {len(df_clean)}')

# Convert Month_Year to datetime and sort
df_clean['Month_Year'] = pd.to_datetime(df_clean['Month_Year'])
df_clean = df_clean.sort_values(['Local Site Name', 'Month_Year'])

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
    x = layers.ConvLSTM2D(32, (3,3), padding='same', return_sequences=True)(x)
    return layers.Add()([shortcut, x])

def create_knn_indices(grid_h, grid_w, k=5):
    x = np.arange(grid_h)
    y = np.arange(grid_w)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    return indices[:, 1:]

def build_ksc_convlstm(knn_indices_1d, grid_h, grid_w, seq_length=6, features=5):
    inputs = layers.Input(shape=(seq_length, grid_h, grid_w, features))
    
    # KNN Spatial Filtering
    x = KNNReducer(knn_indices_1d)(inputs)
    
    # ConvLSTM backbone
    x = layers.ConvLSTM2D(32, (3,3), padding='same', return_sequences=True, activation='tanh')(x)
    
    # Residual blocks with attention
    x = residual_block(x)
    x = SpatioTemporalAttention()(x)
    
    # Final prediction
    x = layers.ConvLSTM2D(32, (3,3), padding='same', return_sequences=False)(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation='relu')(x)
    
    return Model(inputs, outputs)

# Prepare training data
print('\nPreparing training data...')

# Create sequences for each site
def create_sequences(site_data, seq_length=6):
    sequences = []
    targets = []
    
    site_data = site_data.sort_values('Month_Year')
    
    # Fill missing pollutant values with mean
    pollutant_cols = ['PM2.5', 'CO', 'SO2', 'NO', 'PM10']
    for col in pollutant_cols:
        if col in site_data.columns:
            site_data[col] = site_data[col].fillna(site_data[col].mean())
    
    if len(site_data) < seq_length + 1:
        return sequences, targets
    
    for i in range(len(site_data) - seq_length):
        seq_data = site_data.iloc[i:i+seq_length][pollutant_cols].values
        target = site_data.iloc[i+seq_length]['PM2.5']
        
        if not np.isnan(target) and not np.any(np.isnan(seq_data)):
            sequences.append(seq_data)
            targets.append(target)
    
    return sequences, targets

# Create sequences for all sites
all_sequences = []
all_targets = []
site_names = []

for site_name in df_clean['Local Site Name'].unique():
    site_data = df_clean[df_clean['Local Site Name'] == site_name].copy()
    sequences, targets = create_sequences(site_data, seq_length=6)
    
    if sequences:
        all_sequences.extend(sequences)
        all_targets.extend(targets)
        site_names.extend([site_name] * len(sequences))

print(f'Created {len(all_sequences)} training sequences')

if len(all_sequences) == 0:
    print('No valid sequences created. Exiting.')
    exit(1)

# Convert to numpy arrays and normalize
X = np.array(all_sequences)
y = np.array(all_targets)

print(f'Input shape: {X.shape}')
print(f'Target shape: {y.shape}')

# Normalize features
scaler_X = StandardScaler()
X_reshaped = X.reshape(-1, X.shape[-1])
X_scaled = scaler_X.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(X.shape)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

print(f'Training set: {X_train.shape[0]} samples')
print(f'Test set: {X_test.shape[0]} samples')

# Reshape for ConvLSTM (add spatial dimensions)
def reshape_for_convlstm(X, grid_size=10):
    batch_size, seq_len, features = X.shape
    # Create a simple spatial arrangement
    X_spatial = np.zeros((batch_size, seq_len, grid_size, grid_size, features))
    
    # Place data in center of grid
    center = grid_size // 2
    X_spatial[:, :, center, center, :] = X
    
    # Add some spatial variation
    for i in range(1, min(3, grid_size//2)):
        if center-i >= 0:
            X_spatial[:, :, center-i, center, :] = X * 0.9
            X_spatial[:, :, center, center-i, :] = X * 0.9
        if center+i < grid_size:
            X_spatial[:, :, center+i, center, :] = X * 0.9
            X_spatial[:, :, center, center+i, :] = X * 0.9
    
    return X_spatial

GRID_SIZE = 10
X_train_spatial = reshape_for_convlstm(X_train, GRID_SIZE)
X_test_spatial = reshape_for_convlstm(X_test, GRID_SIZE)

print(f'Spatial training shape: {X_train_spatial.shape}')

# Configuration
SEQ_LENGTH = 6
FEATURES = 5
K = 5

print('\nCreating KNN indices...')
knn_1d = create_knn_indices(GRID_SIZE, GRID_SIZE, k=K)

print('Building KSC-ConvLSTM model...')
model = build_ksc_convlstm(
    knn_indices_1d=knn_1d,
    grid_h=GRID_SIZE,
    grid_w=GRID_SIZE,
    seq_length=SEQ_LENGTH,
    features=FEATURES
)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse',
    metrics=['mae']
)

print('Model compiled successfully!')

# Train model
print('\nTraining model...')
history = model.fit(
    X_train_spatial, y_train,
    validation_data=(X_test_spatial, y_test),
    epochs=20,
    batch_size=16,
    verbose=1
)

# Make predictions
print('\nMaking predictions...')
y_pred_scaled = model.predict(X_test_spatial)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate metrics
mae = mean_absolute_error(y_test_actual, y_pred)
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_pred)

print('\n=== PREDICTION SUMMARY ===')
print(f'Test Samples: {len(y_test_actual)}')
print(f'Mean Absolute Error (MAE): {mae:.4f} μg/m³')
print(f'Root Mean Square Error (RMSE): {rmse:.4f} μg/m³')
print(f'R² Score: {r2:.4f}')
print(f'Mean Actual PM2.5: {np.mean(y_test_actual):.4f} μg/m³')
print(f'Mean Predicted PM2.5: {np.mean(y_pred):.4f} μg/m³')

# Prediction statistics
print('\n=== PREDICTION STATISTICS ===')
print(f'Actual PM2.5 Range: {np.min(y_test_actual):.2f} - {np.max(y_test_actual):.2f} μg/m³')
print(f'Predicted PM2.5 Range: {np.min(y_pred):.2f} - {np.max(y_pred):.2f} μg/m³')
print(f'Prediction Std Dev: {np.std(y_pred):.4f} μg/m³')
print(f'Actual Std Dev: {np.std(y_test_actual):.4f} μg/m³')

# Error analysis
errors = np.abs(y_test_actual - y_pred)
print(f'\n=== ERROR ANALYSIS ===')
print(f'Mean Absolute Error: {np.mean(errors):.4f} μg/m³')
print(f'Median Absolute Error: {np.median(errors):.4f} μg/m³')
print(f'90th Percentile Error: {np.percentile(errors, 90):.4f} μg/m³')
print(f'Max Error: {np.max(errors):.4f} μg/m³')

# Model performance by prediction range
low_mask = y_test_actual < np.percentile(y_test_actual, 33)
mid_mask = (y_test_actual >= np.percentile(y_test_actual, 33)) & (y_test_actual < np.percentile(y_test_actual, 67))
high_mask = y_test_actual >= np.percentile(y_test_actual, 67)

print(f'\n=== PERFORMANCE BY PM2.5 LEVELS ===')
print(f'Low PM2.5 (< {np.percentile(y_test_actual, 33):.2f}): MAE = {mean_absolute_error(y_test_actual[low_mask], y_pred[low_mask]):.4f}')
print(f'Medium PM2.5: MAE = {mean_absolute_error(y_test_actual[mid_mask], y_pred[mid_mask]):.4f}')
print(f'High PM2.5 (> {np.percentile(y_test_actual, 67):.2f}): MAE = {mean_absolute_error(y_test_actual[high_mask], y_pred[high_mask]):.4f}')

# Save results
results_df = pd.DataFrame({
    'Actual_PM25': y_test_actual,
    'Predicted_PM25': y_pred,
    'Absolute_Error': errors
})
results_df.to_csv('ksc_model_predictions.csv', index=False)
print(f'\nPredictions saved to ksc_model_predictions.csv')

# Save model
model.save('ksc_convlstm_model.h5')
print('Model saved to ksc_convlstm_model.h5')

print('\n=== KSC-ConvLSTM Training Complete ===')
