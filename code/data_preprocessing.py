import pandas as pd

sp500_data = pd.read_excel('SP500_Monthly_Data.xlsx')
gdp_data = pd.read_excel('GDP According to EDP.xlsx')
inflation_data = pd.read_excel('Inflation Data according to US Bureau.xlsx', skiprows=10)

inflation_data.columns = inflation_data.iloc[0]
inflation_data = inflation_data.drop(0)
inflation_data.columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'HALF1', 'HALF2']
inflation_data = inflation_data.dropna(subset=['Year'])
inflation_data['Year'] = inflation_data['Year'].astype(int)
inflation_data.iloc[:, 1:] = inflation_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

gdp_inflation_merged = pd.merge(gdp_data, inflation_data, on='Year', how='inner')
combined_data = pd.merge(gdp_inflation_merged, sp500_data, on='Year', how='inner')

combined_data['GDP_Lag1'] = combined_data['GDP'].shift(1)
combined_data['Jan_Lag1'] = combined_data['Jan'].shift(1)
combined_data['Close_Lag1'] = combined_data['Close'].shift(1)
combined_data['Jan_MA3'] = combined_data['Jan'].rolling(window=3).mean()
combined_data['Close_MA3'] = combined_data['Close'].rolling(window=3).mean()
combined_data['GDP_Jan_Interaction'] = combined_data['GDP'] * combined_data['Jan']
combined_data = combined_data.dropna()

combined_data.to_csv('data/combined_data.csv', index=False)
