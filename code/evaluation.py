import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

combined_data = pd.read_csv(combined_data.csv)

X = combined_data[['GDP_Lag1', 'Jan_Lag1', 'Close_Lag1', 'Jan_MA3', 'Close_MA3', 'GDP_Jan_Interaction']]
y = combined_data['GDP Growth (%)']

linear_model = joblib.load('models/linear_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
gb_model = joblib.load('models/gb_model.pkl')

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mae, mse, r2

_, mae_linear, mse_linear, r2_linear = evaluate_model(linear_model, X, y)
y_pred_rf, mae_rf, mse_rf, r2_rf = evaluate_model(rf_model, X, y)
_, mae_gb, mse_gb, r2_gb = evaluate_model(gb_model, X, y)

print("Linear Regression: MAE =", mae_linear, ", MSE =", mse_linear, ", R2 =", r2_linear)
print("Random Forest: MAE =", mae_rf, ", MSE =", mse_rf, ", R2 =", r2_rf)
print("Gradient Boosting: MAE =", mae_gb, ", MSE =", mse_gb, ", R2 =", r2_gb)

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig('figures/feature_importance_random_forest.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred_rf, color='blue', edgecolors='k', alpha=0.7)
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', lw=2)
plt.title("Actual vs Predicted GDP Growth (Random Forest)")
plt.xlabel("Actual GDP Growth (%)")
plt.ylabel("Predicted GDP Growth (%)")
plt.tight_layout()
plt.savefig('figures/actual_vs_predicted_gdp_growth.png')
plt.show()
