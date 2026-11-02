"""
Part 2: Time Series Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# =================================================================
# Q1: Preliminary assessment of the raw series
# =================================================================
df = pd.read_csv('data/time-series.csv')
y = df['V4'].copy()
y.index = pd.RangeIndex(1, len(y) + 1)

print("Length:", len(y))
print(y.describe())
print("Skewness:", stats.skew(y))
print("Excess kurtosis:", stats.kurtosis(y))

# Full series plot (Figure 1)
plt.figure(figsize=(14, 4))
plt.plot(y)
plt.title('Full Series V4 (n=24000)')
plt.xlabel('Quarter index')
plt.ylabel('Sales')
plt.tight_layout()
plt.savefig('results/full_series.png', dpi=130)
plt.show()

# Row-wise CV check (rows = time, columns = 8 related products)
sub = df[['V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9']]
cv = sub.std(axis=1) / sub.mean(axis=1)
print("\nRow-wise coefficient of variation (median):", cv.median())

# Trend check: block means over 1,000-period chunks
block_means = y.groupby(np.arange(len(y)) // 1000).mean()
print("\nBlock means (every 1000 obs):\n", block_means)

# ADF test: H0 = unit root (non-stationary), H1 = stationary
adf_stat, adf_p, adf_lags, *_ = adfuller(y, autolag='AIC')
print(f"\nADF statistic: {adf_stat:.4f}, p-value: {adf_p:.4g}, lags used: {adf_lags}")

# ACF / PACF values
acf_vals = acf(y, nlags=20)
pacf_vals = pacf(y, nlags=20)
print("\nACF lags 1-12:", np.round(acf_vals[1:13], 4))
print("PACF lags 1-12:", np.round(pacf_vals[1:13], 4))

# ACF / PACF plot (Figure 2)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
plot_acf(y, ax=axes[0], lags=40)
axes[0].set_title('ACF - V4 (raw)')
plot_pacf(y, ax=axes[1], lags=40, method='ywm')
axes[1].set_title('PACF - V4 (raw)')
plt.tight_layout()
plt.savefig('results/acf_pacf_raw.png', dpi=130)
plt.show()

# Seasonal decomposition, period = 4
decomp = seasonal_decompose(y.iloc[:400], model='additive', period=4)
print("\nSeasonal component range:", decomp.seasonal.min(), decomp.seasonal.max())

# =================================================================
# Q2: Estimating and assessing candidate models
# =================================================================
n_test = 20
train, test = y.iloc[:-n_test], y.iloc[-n_test:]
print("\nTrain length:", len(train), "Test length:", len(test))

results = {}

results['Naive'] = pd.Series([train.iloc[-1]] * n_test, index=test.index)
results['Mean'] = pd.Series([train.mean()] * n_test, index=test.index)

sma3_val = train.iloc[-3:].mean()
results['SMA(3)'] = pd.Series([sma3_val] * n_test, index=test.index)

fit_ses = SimpleExpSmoothing(train).fit()
results['SES'] = fit_ses.forecast(n_test)
print("\nSES smoothing level (alpha):", fit_ses.params['smoothing_level'])

# AR/ARMA models (d=0, series already stationary)
fit_ar1 = ARIMA(train, order=(1, 0, 0)).fit()
results['AR(1)'] = fit_ar1.forecast(n_test)

fit_arma11 = ARIMA(train, order=(1, 0, 1)).fit()
results['ARMA(1,1)'] = fit_arma11.forecast(n_test)

fit_arma21 = ARIMA(train, order=(2, 0, 1)).fit()
results['ARMA(2,1)'] = fit_arma21.forecast(n_test)

fit_arma22 = ARIMA(train, order=(2, 0, 2)).fit()
results['ARMA(2,2)'] = fit_arma22.forecast(n_test)

# Out-of-sample accuracy
print("\n--- Out-of-sample forecast accuracy (test set, n=20) ---")
for name, fc in results.items():
    mae = mean_absolute_error(test, fc)
    rmse = mean_squared_error(test, fc) ** 0.5
    mape = mean_absolute_percentage_error(test, fc) * 100
    print(f"{name:12s}  MAE={mae:9.2f}  RMSE={rmse:9.2f}  MAPE={mape:7.2f}%")

# AIC/BIC and Ljung-Box (H0: residuals are white noise)
print("\n--- In-sample AIC/BIC and Ljung-Box test ---")
for name, fit in [('AR(1)', fit_ar1), ('ARMA(1,1)', fit_arma11),
                   ('ARMA(2,1)', fit_arma21), ('ARMA(2,2)', fit_arma22)]:
    lb = acorr_ljungbox(fit.resid, lags=[10], return_df=True)
    print(f"{name:12s} AIC={fit.aic:.2f} BIC={fit.bic:.2f} "
          f"Ljung-Box(10) p={lb['lb_pvalue'].values[0]:.4f}")

# =================================================================
# Q3: Forecast for the next 4 quarters (best model: SES)
# =================================================================
# Refit SES on the full series before forecasting
final_ses = SimpleExpSmoothing(y).fit()
fc4_ses = final_ses.forecast(4)
print("\nFinal SES alpha:", final_ses.params['smoothing_level'])
print("\nSES 4-quarter forecast:\n", fc4_ses)

# ARMA(2,2) shown for reference/comparison
final_arma22 = ARIMA(y, order=(2, 0, 2)).fit()
fc4_arma22 = final_arma22.forecast(4)
print("\nARMA(2,2) 4-quarter forecast (reference):\n", fc4_arma22)

# Forecast plot (Figure 3)
fig, ax = plt.subplots(figsize=(12, 5))
tail = y.iloc[-60:]
ax.plot(tail.index, tail.values, label='Historical (last 60 quarters)', color='black')
ax.plot(fc4_ses.index, fc4_ses.values, label='SES forecast (next 4 quarters)',
        color='red', marker='o')
ax.plot(fc4_arma22.index, fc4_arma22.values, label='ARMA(2,2) forecast (next 4 quarters)',
        color='green', marker='s', linestyle='--')
ax.axhline(y.mean(), color='gray', linestyle=':', label=f'Overall mean ({y.mean():.0f})')
ax.set_title('V4 — Forecast for Next 4 Quarters')
ax.set_xlabel('Quarter index')
ax.set_ylabel('Sales')
ax.legend()
plt.tight_layout()
plt.savefig('results/forecast_4q.png', dpi=130)
plt.show()
