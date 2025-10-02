import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 8)

print("="*60)
print("SARIMAX FORECASTING - CCTV TRAFFIC")
print("="*60)

# ============================================================
# STEP 1: LOAD DAN EXPLORE DATA
# ============================================================
print("\n[STEP 1] Loading data...")

df = pd.read_csv('dataset/data_preprocessed_v3_full_24h_fixed.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')
df = df.sort_index()

print(f"\nDataset Info:")
print(f"- Total rows: {len(df)}")
print(f"- Date range: {df.index.min()} to {df.index.max()}")
print(f"- Columns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")

print(f"\nDescriptive Statistics:")
print(df[['in', 'out', 'total_traffic']].describe())

# Visualisasi pattern
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

axes[0].plot(df.index, df['in'], label='In', color='blue', linewidth=1.5)
axes[0].set_title('Traffic IN Over Time', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(df.index, df['out'], label='Out', color='red', linewidth=1.5)
axes[1].set_title('Traffic OUT Over Time', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(df.index, df['total_traffic'], label='Total Traffic', color='green', linewidth=1.5)
axes[2].set_title('Total Traffic Over Time', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Count')
axes[2].set_xlabel('Date')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/01_data_exploration.png', dpi=300, bbox_inches='tight')
print("Saved: results/01_data_exploration.png")
plt.close()

# Heatmap by hour and day
pivot_data = df.pivot_table(values='total_traffic', index='hour', columns='day_name', aggfunc='mean')
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot_data = pivot_data.reindex(columns=[d for d in day_order if d in pivot_data.columns])

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Avg Traffic'})
plt.title('Average Traffic Heatmap (Hour vs Day)', fontsize=14, fontweight='bold')
plt.xlabel('Day of Week')
plt.ylabel('Hour of Day')
plt.tight_layout()
plt.savefig('results/01_traffic_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: results/01_traffic_heatmap.png")
plt.close()

print("\n[STEP 1] Completed")

# ============================================================
# STEP 2: FEATURE ENGINEERING (EXOGENOUS VARIABLES)
# ============================================================
print("\n[STEP 2] Feature engineering...")

# Cyclical encoding untuk hour (karena hour 23 dekat dengan hour 0)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Cyclical encoding untuk day_of_week
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Exogenous variables yang akan digunakan
exog_columns = ['hour', 'is_weekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']

print(f"\nExogenous variables selected: {exog_columns}")
print(f"\nSample data with features:")
print(df[['total_traffic'] + exog_columns].head(10))

print("\n[STEP 2] Completed")

# ============================================================
# STEP 3: TEST STATIONARITY
# ============================================================
print("\n[STEP 3] Testing stationarity...")

def test_stationarity(timeseries, title):
    print(f"\n{title}")
    print("-" * 50)

    # ADF Test
    adf_result = adfuller(timeseries.dropna(), autolag='AIC')
    print(f"ADF Test:")
    print(f"  ADF Statistic: {adf_result[0]:.6f}")
    print(f"  p-value: {adf_result[1]:.6f}")
    print(f"  Critical Values:")
    for key, value in adf_result[4].items():
        print(f"    {key}: {value:.3f}")

    if adf_result[1] <= 0.05:
        print(f"  Result: STATIONARY (p < 0.05)")
    else:
        print(f"  Result: NON-STATIONARY (p >= 0.05)")

    # KPSS Test
    kpss_result = kpss(timeseries.dropna(), regression='ct', nlags='auto')
    print(f"\nKPSS Test:")
    print(f"  KPSS Statistic: {kpss_result[0]:.6f}")
    print(f"  p-value: {kpss_result[1]:.6f}")
    print(f"  Critical Values:")
    for key, value in kpss_result[3].items():
        print(f"    {key}: {value:.3f}")

    if kpss_result[1] >= 0.05:
        print(f"  Result: STATIONARY (p >= 0.05)")
    else:
        print(f"  Result: NON-STATIONARY (p < 0.05)")

    return adf_result, kpss_result

# Test untuk total_traffic
adf_res, kpss_res = test_stationarity(df['total_traffic'], "Stationarity Test: Total Traffic")

# Visualisasi
fig, axes = plt.subplots(2, 1, figsize=(15, 8))

axes[0].plot(df.index, df['total_traffic'], linewidth=1.5, color='blue')
axes[0].set_title('Total Traffic Time Series', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Traffic Count')
axes[0].grid(True, alpha=0.3)

axes[1].plot(df.index[1:], df['total_traffic'].diff().dropna(), linewidth=1.5, color='orange')
axes[1].set_title('First Difference of Total Traffic', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Difference')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/03_stationarity_test.png', dpi=300, bbox_inches='tight')
print("\nSaved: results/03_stationarity_test.png")
plt.close()

print("\n[STEP 3] Completed")

# ============================================================
# STEP 4: ACF/PACF PLOT (Parameter Identification)
# ============================================================
print("\n[STEP 4] Identifying SARIMAX parameters...")

# ACF dan PACF plots
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

plot_acf(df['total_traffic'].dropna(), lags=40, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

plot_pacf(df['total_traffic'].dropna(), lags=40, ax=axes[1], method='ywm')
axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/04_acf_pacf.png', dpi=300, bbox_inches='tight')
print("\nSaved: results/04_acf_pacf.png")
plt.close()

print("\n[STEP 4] Completed")
print("\nNote: Karena data kecil, kita akan menggunakan parameter SARIMAX sederhana:")
print("  - order=(1,0,1): AR=1, I=0 (sudah stasioner), MA=1")
print("  - seasonal_order=(1,0,1,24): Seasonal period 24 jam")
print("  - exog: hour, is_weekend, hour_sin, hour_cos, day_sin, day_cos")

# ============================================================
# STEP 5: SPLIT DATA (Train/Test)
# ============================================================
print("\n[STEP 5] Splitting data...")

# Split: 5 hari train (120 jam), 2 hari test (48 jam)
train_size = 120
test_size = len(df) - train_size

train = df[:train_size]
test = df[train_size:]

# Split exogenous variables
exog_train = train[exog_columns]
exog_test = test[exog_columns]

# Split target
y_train = train['total_traffic']
y_test = test['total_traffic']

print(f"\nTrain set: {len(train)} rows ({train.index.min()} to {train.index.max()})")
print(f"Test set: {len(test)} rows ({test.index.min()} to {test.index.max()})")
print(f"\nExogenous variables shape:")
print(f"  Train: {exog_train.shape}")
print(f"  Test: {exog_test.shape}")

print("\n[STEP 5] Completed")

# ============================================================
# STEP 6: TRAIN SARIMAX MODEL
# ============================================================
print("\n[STEP 6] Training SARIMAX model...")
print("This may take a few minutes...")

# Define SARIMAX parameters
order = (1, 0, 1)  # (p, d, q)
seasonal_order = (1, 0, 1, 24)  # (P, D, Q, s)

# Train SARIMAX model
model = SARIMAX(
    y_train,
    exog=exog_train,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Fit model
model_fit = model.fit(disp=False, maxiter=200)

print("\nModel Summary:")
print("="*60)
print(model_fit.summary())

print("\n[STEP 6] Completed")

# ============================================================
# STEP 7: RESIDUAL ANALYSIS
# ============================================================
print("\n[STEP 7] Validating model (residual analysis)...")

# Get residuals
residuals = model_fit.resid

# Plot residuals
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Residuals over time
axes[0, 0].plot(residuals)
axes[0, 0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Index')
axes[0, 0].set_ylabel('Residual')
axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[0, 0].grid(True, alpha=0.3)

# Residuals histogram
axes[0, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Residuals Histogram', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# ACF of residuals
plot_acf(residuals, lags=30, ax=axes[1, 0])
axes[1, 0].set_title('ACF of Residuals', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/07_residual_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: results/07_residual_analysis.png")
plt.close()

print(f"\nResidual Statistics:")
print(f"  Mean: {residuals.mean():.4f}")
print(f"  Std Dev: {residuals.std():.4f}")
print(f"  Min: {residuals.min():.4f}")
print(f"  Max: {residuals.max():.4f}")

print("\n[STEP 7] Completed")

# ============================================================
# STEP 8: FORECAST TEST SET
# ============================================================
print("\n[STEP 8] Forecasting test set...")

# Forecast
forecast_result = model_fit.get_forecast(steps=len(test), exog=exog_test)
y_pred = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

print(f"Forecast shape: {y_pred.shape}")
print(f"Confidence interval shape: {conf_int.shape}")

print("\n[STEP 8] Completed")

# ============================================================
# STEP 9: EVALUATE METRICS
# ============================================================
print("\n[STEP 9] Calculating evaluation metrics...")

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# MAPE with proper handling of zero values
# Convert to numpy arrays to avoid indexing issues
y_test_array = y_test.values
y_pred_array = y_pred.values

# Only calculate MAPE for non-zero actual values
mask = y_test_array > 0
if mask.sum() > 0:
    mape = np.mean(np.abs((y_test_array[mask] - y_pred_array[mask]) / y_test_array[mask])) * 100
else:
    mape = np.nan

r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("MODEL EVALUATION METRICS")
print("="*60)
print(f"MAE (Mean Absolute Error):        {mae:.2f}")
print(f"RMSE (Root Mean Squared Error):   {rmse:.2f}")
print(f"MAPE (Mean Absolute % Error):     {mape:.2f}%")
print(f"R² Score:                          {r2:.4f}")
print("="*60)

print("\n[STEP 9] Completed")

# ============================================================
# STEP 10: VISUALIZE RESULTS
# ============================================================
print("\n[STEP 10] Creating visualizations...")

# Plot 1: Actual vs Predicted (Full data)
fig, ax = plt.subplots(figsize=(18, 6))

ax.plot(train.index, y_train, label='Train', color='blue', linewidth=1.5, alpha=0.7)
ax.plot(test.index, y_test, label='Test (Actual)', color='green', linewidth=2, marker='o', markersize=4)
ax.plot(test.index, y_pred, label='Forecast', color='red', linewidth=2, linestyle='--', marker='s', markersize=4)
ax.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                color='pink', alpha=0.3, label='95% Confidence Interval')

ax.set_title('SARIMAX Forecast: Actual vs Predicted', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Total Traffic', fontsize=12)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/10_forecast_vs_actual.png', dpi=300, bbox_inches='tight')
print("Saved: results/10_forecast_vs_actual.png")
plt.close()

# Plot 2: Test set only (zoomed)
fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(test.index, y_test, label='Actual', color='green', linewidth=2.5, marker='o', markersize=6)
ax.plot(test.index, y_pred, label='Forecast', color='red', linewidth=2.5, linestyle='--', marker='s', markersize=6)
ax.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                color='pink', alpha=0.3, label='95% Confidence Interval')

ax.set_title('Test Set: Actual vs Forecast (Zoomed)', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Total Traffic', fontsize=12)
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

# Add metrics text box
textstr = f'MAE: {mae:.2f}\\nRMSE: {rmse:.2f}\\nMAPE: {mape:.2f}%\\nR²: {r2:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('results/10_test_forecast_zoomed.png', dpi=300, bbox_inches='tight')
print("Saved: results/10_test_forecast_zoomed.png")
plt.close()

print("\n[STEP 10] Completed")

# ============================================================
# STEP 11: FORECAST FUTURE (7 days ahead)
# ============================================================
print("\n[STEP 11] Forecasting future (7 days)...")

# Create future dates (7 days = 168 hours)
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=168, freq='h')

# Create future exogenous variables
future_exog = pd.DataFrame(index=future_dates)
future_exog['hour'] = future_dates.hour
future_exog['day_of_week'] = future_dates.dayofweek
future_exog['is_weekend'] = future_exog['day_of_week'].isin([5, 6]).astype(int)
future_exog['hour_sin'] = np.sin(2 * np.pi * future_exog['hour'] / 24)
future_exog['hour_cos'] = np.cos(2 * np.pi * future_exog['hour'] / 24)
future_exog['day_sin'] = np.sin(2 * np.pi * future_exog['day_of_week'] / 7)
future_exog['day_cos'] = np.cos(2 * np.pi * future_exog['day_of_week'] / 7)
future_exog = future_exog[exog_columns]

# Forecast future using the model fitted on ALL data
print("Retraining model on full dataset...")
model_full = SARIMAX(
    df['total_traffic'],
    exog=df[exog_columns],
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
model_full_fit = model_full.fit(disp=False, maxiter=200)

# Forecast
future_forecast = model_full_fit.get_forecast(steps=168, exog=future_exog)
future_pred = future_forecast.predicted_mean
future_conf_int = future_forecast.conf_int()

print(f"Future forecast shape: {future_pred.shape}")

# Save future forecast to CSV
future_results = pd.DataFrame({
    'datetime': future_dates,
    'predicted_traffic': future_pred.values,
    'lower_ci': future_conf_int.iloc[:, 0].values,
    'upper_ci': future_conf_int.iloc[:, 1].values,
    'hour': future_exog['hour'].values,
    'is_weekend': future_exog['is_weekend'].values
})
future_results.to_csv('results/future_forecast_7days.csv', index=False)
print("Saved: results/future_forecast_7days.csv")

# Plot future forecast
fig, ax = plt.subplots(figsize=(18, 6))

ax.plot(df.index, df['total_traffic'], label='Historical Data', color='blue', linewidth=1.5, alpha=0.7)
ax.plot(future_dates, future_pred, label='Future Forecast (7 days)', color='red', linewidth=2, linestyle='--')
ax.fill_between(future_dates, future_conf_int.iloc[:, 0], future_conf_int.iloc[:, 1],
                color='pink', alpha=0.3, label='95% Confidence Interval')

ax.axvline(x=df.index[-1], color='black', linestyle=':', linewidth=2, label='Forecast Start')
ax.set_title('7-Day Future Traffic Forecast', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Total Traffic', fontsize=12)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/11_future_forecast_7days.png', dpi=300, bbox_inches='tight')
print("Saved: results/11_future_forecast_7days.png")
plt.close()

print("\n[STEP 11] Completed")

# ============================================================
# STEP 12: SAVE MODEL
# ============================================================
print("\n[STEP 12] Saving model...")

import pickle

# Save model
with open('results/sarimax_model.pkl', 'wb') as f:
    pickle.dump(model_full_fit, f)
print("Saved: results/sarimax_model.pkl")

# Save test results
test_results = pd.DataFrame({
    'datetime': test.index,
    'actual': y_test.values,
    'predicted': y_pred.values,
    'lower_ci': conf_int.iloc[:, 0].values,
    'upper_ci': conf_int.iloc[:, 1].values
})
test_results.to_csv('results/test_predictions.csv', index=False)
print("Saved: results/test_predictions.csv")

# Save metrics
metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'MAPE (%)', 'R2'],
    'Value': [mae, rmse, mape, r2]
})
metrics_df.to_csv('results/evaluation_metrics.csv', index=False)
print("Saved: results/evaluation_metrics.csv")

print("\n[STEP 12] Completed")

print("\n" + "="*60)
print("SARIMAX FORECASTING COMPLETED!")
print("="*60)
print("\nResults saved in 'results/' directory:")
print("  - Visualizations: PNG files")
print("  - Model: sarimax_model.pkl")
print("  - Test predictions: test_predictions.csv")
print("  - Future forecast: future_forecast_7days.csv")
print("  - Metrics: evaluation_metrics.csv")
print("="*60)
