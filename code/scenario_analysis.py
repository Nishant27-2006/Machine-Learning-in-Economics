import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load the processed data and the trained Random Forest model
combined_data = pd.read_csv('combined_data.csv')
rf_model = joblib.load('rf_model.pkl')

# Selecting features for the analysis
X_test = combined_data[['GDP_Lag1', 'Jan_Lag1', 'Close_Lag1', 'Jan_MA3', 'Close_MA3', 'GDP_Jan_Interaction']]

# Define scenarios by adjusting the inflation rate and S&P 500 growth
# We'll use the median of the test set as the base for inflation and S&P 500 growth

# Extract median values from the test set
median_inflation = X_test['Jan_Lag1'].median()
median_sp500_growth = X_test['Close_Lag1'].median()

# Scenario 1: High Inflation, Low S&P 500 Growth
scenario_1 = X_test.copy()
scenario_1['Jan_Lag1'] = median_inflation * 1.5  # 50% increase in inflation
scenario_1['Close_Lag1'] = median_sp500_growth * 0.8  # 20% decrease in S&P 500 growth

# Scenario 2: Low Inflation, High S&P 500 Growth
scenario_2 = X_test.copy()
scenario_2['Jan_Lag1'] = median_inflation * 0.8  # 20% decrease in inflation
scenario_2['Close_Lag1'] = median_sp500_growth * 1.5  # 50% increase in S&P 500 growth

# Scenario 3: Baseline (Current) Conditions
scenario_3 = X_test.copy()

# Predict GDP Growth for each scenario using the Random Forest model
scenario_1_pred = rf_model.predict(scenario_1)
scenario_2_pred = rf_model.predict(scenario_2)
scenario_3_pred = rf_model.predict(scenario_3)

scenario_predictions = pd.DataFrame({
    'Scenario 1: High Inflation, Low S&P Growth': scenario_1_pred,
    'Scenario 2: Low Inflation, High S&P Growth': scenario_2_pred,
    'Scenario 3: Baseline Conditions': scenario_3_pred
})

scenario_1_mean = np.mean(scenario_1_pred)
scenario_2_mean = np.mean(scenario_2_pred)
scenario_3_mean = np.mean(scenario_3_pred)

scenarios = ['High Inflation, Low S&P Growth', 'Low Inflation, High S&P Growth', 'Baseline Conditions']
predictions = [scenario_1_mean, scenario_2_mean, scenario_3_mean]
plt.figure(figsize=(10, 6))
plt.bar(scenarios, predictions, color=['red', 'green', 'blue'])
plt.title('GDP Growth Predictions Under Different Economic Scenarios')
plt.xlabel('Scenario')
plt.ylabel('Predicted GDP Growth (%)')
plt.ylim(min(predictions) - 0.02, max(predictions) + 0.02)  # Adjusting the y-axis limits for better visibility
plt.tight_layout()
plt.savefig('figures/scenario_analysis_gdp_predictions.png')
plt.show()
