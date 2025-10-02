import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 8)

print("="*70)
print("MULTIVARIATE SARIMAX FORECASTING - IN & OUT TRAFFIC")
print("="*70)

# ============================================================
# STEP 1: LOAD DATA & FEATURE ENGINEERING
# ============================================================
print("\n[STEP 1] Loading data and creating features...")

df = pd.read_csv('dataset/data_preprocessed_v3_full_24h_fixed.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')
df = df.sort_index()

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

exog_columns = ['hour', 'is_weekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']

print(f"Total rows: {len(df)}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Exogenous variables: {exog_columns}")

# Split data (5 days train, 2 days test)
train_size = 120
train = df[:train_size]
test = df[train_size:]

exog_train = train[exog_columns]
exog_test = test[exog_columns]

print(f"\nTrain: {len(train)} rows | Test: {len(test)} rows")

# ============================================================
# STEP 2: TRAIN MODEL FOR 'IN'
# ============================================================
print("\n[STEP 2] Training SARIMAX model for 'IN' traffic...")

order = (1, 0, 1)
seasonal_order = (1, 0, 1, 24)

# Model for IN
model_in = SARIMAX(
    train['in'],
    exog=exog_train,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
model_in_fit = model_in.fit(disp=False, maxiter=200)

print("Model IN trained successfully")
print(f"  AIC: {model_in_fit.aic:.2f}")
print(f"  BIC: {model_in_fit.bic:.2f}")

# ============================================================
# STEP 3: TRAIN MODEL FOR 'OUT'
# ============================================================
print("\n[STEP 3] Training SARIMAX model for 'OUT' traffic...")

# Model for OUT
model_out = SARIMAX(
    train['out'],
    exog=exog_train,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
model_out_fit = model_out.fit(disp=False, maxiter=200)

print("Model OUT trained successfully")
print(f"  AIC: {model_out_fit.aic:.2f}")
print(f"  BIC: {model_out_fit.bic:.2f}")

# ============================================================
# STEP 4: FORECAST TEST SET
# ============================================================
print("\n[STEP 4] Forecasting test set...")

# Forecast IN
forecast_in = model_in_fit.get_forecast(steps=len(test), exog=exog_test)
pred_in = forecast_in.predicted_mean
conf_in = forecast_in.conf_int()

# Forecast OUT
forecast_out = model_out_fit.get_forecast(steps=len(test), exog=exog_test)
pred_out = forecast_out.predicted_mean
conf_out = forecast_out.conf_int()

# Calculate TOTAL
pred_total = pred_in + pred_out

print(f"Forecasts generated:")
print(f"  IN shape: {pred_in.shape}")
print(f"  OUT shape: {pred_out.shape}")
print(f"  TOTAL shape: {pred_total.shape}")

# ============================================================
# STEP 5: EVALUATE METRICS
# ============================================================
print("\n[STEP 5] Evaluating metrics...")

# Metrics for IN
mae_in = mean_absolute_error(test['in'], pred_in)
rmse_in = np.sqrt(mean_squared_error(test['in'], pred_in))
r2_in = r2_score(test['in'], pred_in)

# Metrics for OUT
mae_out = mean_absolute_error(test['out'], pred_out)
rmse_out = np.sqrt(mean_squared_error(test['out'], pred_out))
r2_out = r2_score(test['out'], pred_out)

# Metrics for TOTAL
mae_total = mean_absolute_error(test['total_traffic'], pred_total)
rmse_total = np.sqrt(mean_squared_error(test['total_traffic'], pred_total))
r2_total = r2_score(test['total_traffic'], pred_total)

print("\n" + "="*70)
print("MODEL EVALUATION METRICS")
print("="*70)
print(f"\n{'Metric':<20} {'IN':<15} {'OUT':<15} {'TOTAL':<15}")
print("-"*70)
print(f"{'MAE':<20} {mae_in:<15.2f} {mae_out:<15.2f} {mae_total:<15.2f}")
print(f"{'RMSE':<20} {rmse_in:<15.2f} {rmse_out:<15.2f} {rmse_total:<15.2f}")
print(f"{'R² Score':<20} {r2_in:<15.4f} {r2_out:<15.4f} {r2_total:<15.4f}")
print("="*70)

# ============================================================
# STEP 6: VISUALIZATIONS
# ============================================================
print("\n[STEP 6] Creating visualizations...")

# Plot 1: IN - Actual vs Predicted
fig, axes = plt.subplots(3, 1, figsize=(18, 14))

# IN
axes[0].plot(train.index, train['in'], label='Train', color='blue', linewidth=1.5, alpha=0.7)
axes[0].plot(test.index, test['in'], label='Test (Actual)', color='green', linewidth=2, marker='o', markersize=4)
axes[0].plot(test.index, pred_in, label='Forecast', color='red', linewidth=2, linestyle='--', marker='s', markersize=4)
axes[0].fill_between(test.index, conf_in.iloc[:, 0], conf_in.iloc[:, 1],
                      color='pink', alpha=0.3, label='95% CI')
axes[0].set_title('IN Traffic: Actual vs Forecast', fontsize=14, fontweight='bold')
axes[0].set_ylabel('IN Count')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)
textstr_in = f'MAE: {mae_in:.2f}\nRMSE: {rmse_in:.2f}\nR²: {r2_in:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
axes[0].text(0.02, 0.98, textstr_in, transform=axes[0].transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

# OUT
axes[1].plot(train.index, train['out'], label='Train', color='blue', linewidth=1.5, alpha=0.7)
axes[1].plot(test.index, test['out'], label='Test (Actual)', color='green', linewidth=2, marker='o', markersize=4)
axes[1].plot(test.index, pred_out, label='Forecast', color='red', linewidth=2, linestyle='--', marker='s', markersize=4)
axes[1].fill_between(test.index, conf_out.iloc[:, 0], conf_out.iloc[:, 1],
                      color='pink', alpha=0.3, label='95% CI')
axes[1].set_title('OUT Traffic: Actual vs Forecast', fontsize=14, fontweight='bold')
axes[1].set_ylabel('OUT Count')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)
textstr_out = f'MAE: {mae_out:.2f}\nRMSE: {rmse_out:.2f}\nR²: {r2_out:.4f}'
axes[1].text(0.02, 0.98, textstr_out, transform=axes[1].transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

# TOTAL
axes[2].plot(train.index, train['total_traffic'], label='Train', color='blue', linewidth=1.5, alpha=0.7)
axes[2].plot(test.index, test['total_traffic'], label='Test (Actual)', color='green', linewidth=2, marker='o', markersize=4)
axes[2].plot(test.index, pred_total, label='Forecast (IN+OUT)', color='red', linewidth=2, linestyle='--', marker='s', markersize=4)
axes[2].set_title('TOTAL Traffic: Actual vs Forecast (IN + OUT)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('TOTAL Count')
axes[2].legend(loc='best')
axes[2].grid(True, alpha=0.3)
textstr_total = f'MAE: {mae_total:.2f}\nRMSE: {rmse_total:.2f}\nR²: {r2_total:.4f}'
axes[2].text(0.02, 0.98, textstr_total, transform=axes[2].transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('results/multivariate_test_forecast.png', dpi=300, bbox_inches='tight')
print("Saved: results/multivariate_test_forecast.png")
plt.close()

# Plot 2: Test Set Zoomed (Side by Side)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# IN
axes[0].plot(test.index, test['in'], label='Actual', color='green', linewidth=2.5, marker='o', markersize=6)
axes[0].plot(test.index, pred_in, label='Forecast', color='red', linewidth=2.5, linestyle='--', marker='s', markersize=6)
axes[0].set_title('IN Traffic - Test Set', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# OUT
axes[1].plot(test.index, test['out'], label='Actual', color='green', linewidth=2.5, marker='o', markersize=6)
axes[1].plot(test.index, pred_out, label='Forecast', color='red', linewidth=2.5, linestyle='--', marker='s', markersize=6)
axes[1].set_title('OUT Traffic - Test Set', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Count')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

# TOTAL
axes[2].plot(test.index, test['total_traffic'], label='Actual', color='green', linewidth=2.5, marker='o', markersize=6)
axes[2].plot(test.index, pred_total, label='Forecast', color='red', linewidth=2.5, linestyle='--', marker='s', markersize=6)
axes[2].set_title('TOTAL Traffic - Test Set', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Count')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/multivariate_test_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: results/multivariate_test_comparison.png")
plt.close()

# ============================================================
# STEP 7: FORECAST FUTURE (7 DAYS)
# ============================================================
print("\n[STEP 7] Forecasting future (7 days)...")

# Retrain on full dataset
print("Retraining models on full dataset...")

model_in_full = SARIMAX(
    df['in'],
    exog=df[exog_columns],
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
model_in_full_fit = model_in_full.fit(disp=False, maxiter=200)

model_out_full = SARIMAX(
    df['out'],
    exog=df[exog_columns],
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
model_out_full_fit = model_out_full.fit(disp=False, maxiter=200)

# Create future exogenous variables
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=168, freq='h')

future_exog = pd.DataFrame(index=future_dates)
future_exog['hour'] = future_dates.hour
future_exog['day_of_week'] = future_dates.dayofweek
future_exog['is_weekend'] = future_exog['day_of_week'].isin([5, 6]).astype(int)
future_exog['hour_sin'] = np.sin(2 * np.pi * future_exog['hour'] / 24)
future_exog['hour_cos'] = np.cos(2 * np.pi * future_exog['hour'] / 24)
future_exog['day_sin'] = np.sin(2 * np.pi * future_exog['day_of_week'] / 7)
future_exog['day_cos'] = np.cos(2 * np.pi * future_exog['day_of_week'] / 7)
future_exog = future_exog[exog_columns]

# Forecast
future_in = model_in_full_fit.get_forecast(steps=168, exog=future_exog)
future_in_pred = future_in.predicted_mean
future_in_conf = future_in.conf_int()

future_out = model_out_full_fit.get_forecast(steps=168, exog=future_exog)
future_out_pred = future_out.predicted_mean
future_out_conf = future_out.conf_int()

future_total_pred = future_in_pred + future_out_pred

print(f"Future forecast completed: {len(future_in_pred)} hours")

# Save to CSV
future_results = pd.DataFrame({
    'datetime': future_dates,
    'predicted_in': future_in_pred.values,
    'predicted_out': future_out_pred.values,
    'predicted_total': future_total_pred.values,
    'in_lower_ci': future_in_conf.iloc[:, 0].values,
    'in_upper_ci': future_in_conf.iloc[:, 1].values,
    'out_lower_ci': future_out_conf.iloc[:, 0].values,
    'out_upper_ci': future_out_conf.iloc[:, 1].values,
    'hour': future_exog['hour'].values,
    'is_weekend': future_exog['is_weekend'].values
})
future_results.to_csv('results/multivariate_future_forecast_7days.csv', index=False)
print("Saved: results/multivariate_future_forecast_7days.csv")

# Plot future forecast
fig, axes = plt.subplots(3, 1, figsize=(18, 14))

# IN
axes[0].plot(df.index, df['in'], label='Historical', color='blue', linewidth=1.5, alpha=0.7)
axes[0].plot(future_dates, future_in_pred, label='Forecast (7 days)', color='red', linewidth=2, linestyle='--')
axes[0].fill_between(future_dates, future_in_conf.iloc[:, 0], future_in_conf.iloc[:, 1],
                      color='pink', alpha=0.3, label='95% CI')
axes[0].axvline(x=df.index[-1], color='black', linestyle=':', linewidth=2, label='Forecast Start')
axes[0].set_title('7-Day Future Forecast: IN Traffic', fontsize=14, fontweight='bold')
axes[0].set_ylabel('IN Count')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)

# OUT
axes[1].plot(df.index, df['out'], label='Historical', color='blue', linewidth=1.5, alpha=0.7)
axes[1].plot(future_dates, future_out_pred, label='Forecast (7 days)', color='red', linewidth=2, linestyle='--')
axes[1].fill_between(future_dates, future_out_conf.iloc[:, 0], future_out_conf.iloc[:, 1],
                      color='pink', alpha=0.3, label='95% CI')
axes[1].axvline(x=df.index[-1], color='black', linestyle=':', linewidth=2, label='Forecast Start')
axes[1].set_title('7-Day Future Forecast: OUT Traffic', fontsize=14, fontweight='bold')
axes[1].set_ylabel('OUT Count')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)

# TOTAL
axes[2].plot(df.index, df['total_traffic'], label='Historical', color='blue', linewidth=1.5, alpha=0.7)
axes[2].plot(future_dates, future_total_pred, label='Forecast (7 days)', color='red', linewidth=2, linestyle='--')
axes[2].axvline(x=df.index[-1], color='black', linestyle=':', linewidth=2, label='Forecast Start')
axes[2].set_title('7-Day Future Forecast: TOTAL Traffic (IN + OUT)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('TOTAL Count')
axes[2].legend(loc='best')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/multivariate_future_forecast.png', dpi=300, bbox_inches='tight')
print("Saved: results/multivariate_future_forecast.png")
plt.close()

# ============================================================
# STEP 8: SAVE TEST RESULTS
# ============================================================
print("\n[STEP 8] Saving test results...")

test_results = pd.DataFrame({
    'datetime': test.index,
    'actual_in': test['in'].values,
    'predicted_in': pred_in.values,
    'actual_out': test['out'].values,
    'predicted_out': pred_out.values,
    'actual_total': test['total_traffic'].values,
    'predicted_total': pred_total.values
})
test_results.to_csv('results/multivariate_test_predictions.csv', index=False)
print("Saved: results/multivariate_test_predictions.csv")

# Save metrics
metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R2'],
    'IN': [mae_in, rmse_in, r2_in],
    'OUT': [mae_out, rmse_out, r2_out],
    'TOTAL': [mae_total, rmse_total, r2_total]
})
metrics_df.to_csv('results/multivariate_metrics.csv', index=False)
print("Saved: results/multivariate_metrics.csv")

print("\n" + "="*70)
print("MULTIVARIATE SARIMAX FORECASTING COMPLETED!")
print("="*70)
print("\nResults saved in 'results/' directory:")
print("  - multivariate_test_forecast.png")
print("  - multivariate_test_comparison.png")
print("  - multivariate_future_forecast.png")
print("  - multivariate_test_predictions.csv")
print("  - multivariate_future_forecast_7days.csv")
print("  - multivariate_metrics.csv")
print("="*70)
