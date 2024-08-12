import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

combined_data = pd.read_csv('data/combined_data.csv')

X = combined_data[['GDP_Lag1', 'Jan_Lag1', 'Close_Lag1', 'Jan_MA3', 'Close_MA3', 'GDP_Jan_Interaction']]
y = combined_data['GDP Growth (%)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

joblib.dump(linear_model, 'models/linear_model.pkl')
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(gb_model, 'models/gb_model.pkl')
