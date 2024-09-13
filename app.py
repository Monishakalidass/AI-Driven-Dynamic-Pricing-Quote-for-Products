import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from predict_low import predict_dollar_price_per_unit 

# Load your dataset
excel_file_path = 'C:\\Users\\HP\\Downloads\\AI Driven Dynamic Pricing Quote for Products\\PYTHON\\Commodity_Purchase_Dataset.xls'

df = pd.read_excel(excel_file_path, sheet_name="Sample Data v1.1")

# Encode categorical variables
le_commodity = LabelEncoder()
le_country = LabelEncoder()
le_exchange = LabelEncoder()

df['Product Commodity'] = le_commodity.fit_transform(df['Product Commodity'])
df['Country'] = le_country.fit_transform(df['Country'])
df['Exchange Name'] = le_exchange.fit_transform(df['Exchange Name'])

# Calculate the sales trend (difference between consecutive 'position' values)
df['Sales_Trend'] = df.groupby('Product Commodity')['position'].transform(lambda x: x.diff().fillna(0))

# Define the feature matrix X and target variable y
X = df[['Product Commodity', 'Country', 'position', 'Exchange Name', 'Sales_Trend']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the RandomForest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Title of the app
st.title("AI Driven Dynamic Pricing Quote for Products")

# Introduction
st.write("""
This simple app predicts dynamic pricing system across customer's commodities based on Dollar Price and Metric Ton positions (volumes).
""")

# Dropdown menus for inputs
product_commodity_options = le_commodity.inverse_transform(df['Product Commodity'].unique())
exchange_options = le_exchange.inverse_transform(df['Exchange Name'].unique())
country_options = le_country.inverse_transform(df['Country'].unique())


product_commodity = st.selectbox("Select Commodity Name:", product_commodity_options)
exchange_name = st.selectbox("Select Exchange Name:", exchange_options)
country = st.selectbox("Select Country:", country_options)

# Function to predict and quote price based on sales trend
def quote_price_based_on_sales(commodity_name, exchange_name, country):
    # Encode the inputs
    commodity_encoded = le_commodity.transform([commodity_name])[0]
    exchange_encoded = le_exchange.transform([exchange_name])[0]
    country_encoded = le_country.transform([country])[0]

    # Filter the dataset to find the relevant entries for the given commodity, exchange, and country
    filtered_df = df[(df['Product Commodity'] == commodity_encoded) & 
                     (df['Exchange Name'] == exchange_encoded) & 
                     (df['Country'] == country_encoded)]

    if filtered_df.empty:
        return None, "No data available for the given inputs."

     # Predict prices for each entry using the sales trend
    predictions = rf_model.predict(filtered_df[['Product Commodity', 'Country', 'position', 'Exchange Name', 'Sales_Trend']])

    # Calculate the profit margin for each predicted price
    profit_margins = []
    for i in range(len(predictions)):
        predicted_price = predictions[i]
        actual_price = filtered_df.iloc[i]['price']
        sales_trend = filtered_df.iloc[i]['Sales_Trend']

        if sales_trend > 0:
            adjusted_price = predicted_price * 1.05  # 5% increase
            profit_margin = (adjusted_price - actual_price) / actual_price
        elif sales_trend < 0:
            adjusted_price = predicted_price * 0.95  # 5% decrease
            profit_margin = (actual_price - adjusted_price) / actual_price
        else:
            adjusted_price = predicted_price
            profit_margin = 0  # Stable market, no profit margin adjustment

        profit_margins.append((profit_margin, adjusted_price, i))

    # Find the entry with the highest profit margin
    max_profit_margin, adjusted_price, max_index = max(profit_margins, key=lambda x: x[0])

    # Determine the appropriate message based on the profit margin
    if max_profit_margin > 0:
        message = f"This quoted price will maximize profit by {max_profit_margin * 100:.2f}%."
    elif max_profit_margin < 0:
        message = f"This quoted price will prevent a loss of {-max_profit_margin * 100:.2f}%."
    else:
        message = "Sales are stable. No change in pricing is needed."


    return adjusted_price, message

# Button to trigger prediction
if st.button("Get Price Quote"):
    adjusted_price, message = quote_price_based_on_sales(product_commodity, exchange_name, country)
    
    if adjusted_price is not None:
        st.title(f"The quoted price for the selected commodity is ${adjusted_price:.2f}.")
        st.write(message)
    else:
        st.write(message)  # Display error message or no data available

if st.button("Get Lower price product"):
    predicted_price, contract_number, country, exchange = predict_dollar_price_per_unit(product_commodity)
    
    if predicted_price is not None:
        st.title(f"The predicted lowest dollar price per unit for {product_commodity} is {predicted_price:.2f}, with contract number {contract_number} at {exchange} exchange, from the country {country}.")
        # st.write(message)
    else:
        st.write("errror")  # Display error message or no data available

# Footer
st.write("Created By TEAM - 6!")