import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset (replace with your actual file path)
data = """Date,Open,High,Low,Close
09/26/2024,"26,005.40","26,250.90","25,998.40","26,216.05"
09/25/2024,"25,921.45","26,032.80","25,871.35","26,004.15"
09/24/2024,"25,921.45","26,011.55","25,886.85","25,940.40"
...
"""  # Truncated for brevity; include the entire dataset in practice.

# Load the data into a DataFrame
from io import StringIO

df = pd.read_csv(StringIO(data))


# Data Cleaning: Convert columns to numeric and remove commas
for column in ['Open', 'High', 'Low', 'Close']:
    df[column] = pd.to_numeric(df[column].str.replace(',', ''), errors='coerce')

# Prepare the data
df['Next_Open'] = df['Open'].shift(-1)  # Next day's opening price

# Drop rows with NaN values after shifting
df.dropna(inplace=True)

features = df[['Open', 'High', 'Low', 'Close']]  # Features
labels = df['Next_Open']  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features.values.astype(float), labels.values.astype(float), test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train.reshape(-1 , 4), y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

new_data = np.array([[26005., 26250., 25998., 26216.]]) 
# Low and Close (current day)
predicted_opening_price = model.predict(new_data)
print(f'Predicted Next Day Opening Price: {predicted_opening_price[0]}')




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset (replace with your actual file path)
data = """Date,Open,High,Low,Close
09/26/2024,"26,005.40","26,250.90","25,998.40","26,216.05"
09/25/2024,"25,921.45","26,032.80","25,871.35","26,004.15"
09/24/2024,"25,921.45","26,011.55","25,886.85","25,940.40"
09/23/2024,"25,872.55","25,956.00","25,847.35","25,939.05"
09/20/2024,"25,525.95","25,849.25","25,426.60","25,790.95"
09/19/2024,"25,487.05","25,611.95","25,376.05","25,415.80"
09/18/2024,"25,402.40","25,482.20","25,285.55","25,377.55"
09/17/2024,"25,416.90","25,441.65","25,352.25","25,418.55"
09/16/2024,"25,406.65","25,445.70","25,336.20","25,383.75"
09/13/2024,"25,430.45","25,430.50","25,292.45","25,356.50"
09/12/2024,"25,059.65","25,433.35","24,941.45","25,388.90"
09/11/2024,"25,034.00","25,113.70","24,885.15","24,918.45"
09/10/2024,"24,999.40","25,130.50","24,896.80","25,041.10"
09/09/2024,"24,823.40","24,957.50","24,753.15","24,936.40"
09/06/2024,"25,093.70","25,168.75","24,801.30","24,852.15"
09/05/2024,"25,250.50","25,275.45","25,127.75","25,145.10"
09/04/2024,"25,089.95","25,216.00","25,083.80","25-198-70" 
"""  # Truncated for brevity; include the entire dataset in practice.

# Load the data into a DataFrame
from io import StringIO

df = pd.read_csv(StringIO(data))

# Data Cleaning: Convert columns to numeric and remove commas
for column in ['Open', 'High', 'Low', 'Close']:
    df[column] = pd.to_numeric(df[column].str.replace(',', ''), errors='coerce')

# Prepare the data
df['Next_Open'] = df['Open'].shift(-1)  # Next day's opening price

# Drop rows with NaN values after shifting
df.dropna(inplace=True)

features = df[['Open', 'High', 'Low', 'Close']]  # Features
labels = df['Next_Open']  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features.values.astype(float), labels.values.astype(float), test_size=0.2,
                                                    random_state=42)

# Train the model using Random Forest Regressor
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train.reshape(-1 , 4), y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model without slicing y_test
mse = mean_squared_error(y_test[:len(predictions)], predictions)  # Ensure lengths match
print(f'Mean Squared Error: {mse}')

# Example of predicting next day's opening price from a new input
new_data = np.array([[26005., 26250., 25998., 26216.]])  # Example input for Open (current day), High,
# Low and Close (current day)
predicted_opening_price = model.predict(new_data)
print(f'Predicted Next Day Opening Price: {predicted_opening_price[0]}')




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load dataset (replace with your actual file path)
data = """Date,Open,High,Low,Close
09/26/2024,"26,005.40","26,250.90","25,998.40","26,216.05"
09/25/2024,"25,921.45","26,032.80","25,871.35","26,004.15"
09/24/2024,"25,921.45","26,011.55","25,886.85","25,940.40"
09/23/2024,"25,872.55","25,956.00","25,847.35","25,939.05"
09/20/2024,"25,525.95","25,849.25","25,426.60","25,790.95"
09/19/2024,"25,487.05","25,611.95","25,376.05","25,415.80"
09/18/2024,"25,402.40","25,482.20","25,285.55","25,377.55"
09/17/2024,"25,416.90","25,441.65","25,352.25","25,418.55"
09/16/2024,"25,406.65","25,445.70","25,336.20","25,383.75"
09/13/2024,"25,430.45","25,430.50","25,292.45","25,356.50"
09/12/2024,"25,059.65","25,433.35","24,941.45","25-388-90"
09/11/2024,"25-034-00","  25113-70", "24885-15", "24918-45" 
"""  # Truncated for brevity; include the entire dataset in practice.

# Load the data into a DataFrame
from io import StringIO

df = pd.read_csv(StringIO(data))

# Data Cleaning: Convert columns to numeric and remove commas
for column in ['Open', 'High', 'Low', 'Close']:
    df[column] = pd.to_numeric(df[column].str.replace(',', ''), errors='coerce')

# Prepare the data
df['Next_Open'] = df['Open'].shift(-1)  # Next day's opening price

# Drop rows with NaN values after shifting
df.dropna(inplace=True)

features = df[['Open', 'High', 'Low', 'Close']]  # Features
labels = df['Next_Open']  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features.values.astype(float), labels.values.astype(float), test_size=0.2,
                                                    random_state=42)

# Train the model using Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=100)
model.fit(X_train.reshape(-1 , 4), y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model without slicing y_test
mse = mean_squared_error(y_test[:len(predictions)], predictions)  # Ensure lengths match
print(f'Mean Squared Error: {mse}')

# Example of predicting next day's opening price from a new input
new_data = np.array([[26005., 26250., 25998., 26216.]])  # Example input for Open (current day), High,
# Low and Close (current day)
predicted_opening_price = model.predict(new_data)
print(f'Predicted Next Day Opening Price: {predicted_opening_price[0]}')
