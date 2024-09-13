import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# *Loading the Data*

# excel_file_path = 'E:/CTS/Commodity_Purchase_Dataset.xls'
excel_file_path = 'C:\\Users\\HP\\Downloads\\AI Driven Dynamic Pricing Quote for Products\\PYTHON\\Commodity_Purchase_Dataset.xls'

df = pd.read_excel(excel_file_path, sheet_name="Sample Data v1.2")


# *Data Pre-processing*

le_commodity = LabelEncoder()
le_country = LabelEncoder()
le_exchange = LabelEncoder()

df['Commodity Name'] = le_commodity.fit_transform(df['Commodity Name'])
df['Country'] = le_country.fit_transform(df['Country'])
df['Exchange Name'] = le_exchange.fit_transform(df['Exchange Name'])

# *Defining Features and Target*

X = df[['Commodity Name', 'Country', 'position', 'Exchange Name']]
y = df['price']

# *Split the data into training and testing sets*

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor model

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# RF Model - Prediction function

def predict_dollar_price_per_unit(commodity_name):
    # Encode the commodity name
    commodity_encoded = le_commodity.transform([commodity_name])[0]

    # Filter the dataset to find the relevant entries for the given commodity
    commodity_df = df[df['Commodity Name'] == commodity_encoded]

    # Predict prices for each entry
    predictions = rf_model.predict(commodity_df[['Commodity Name', 'Country', 'position', 'Exchange Name']])

    # Find the index of the minimum predicted price
    min_index = np.argmin(predictions)

    # Retrieve the contract number, predicted price, country and exchange for the lowest priced commodity
    contract_number = commodity_df.iloc[min_index]['Contract number']
    predicted_price = predictions[min_index]
    country = commodity_df.iloc[min_index]['Country']
    exchange = commodity_df.iloc[min_index]['Exchange Name']

    # Decode the country, exchange back to its original label
    country_decoded = le_country.inverse_transform([country])[0]
    exchange_decoded = le_exchange.inverse_transform([exchange])[0]

    return predicted_price, contract_number, country_decoded, exchange_decoded

