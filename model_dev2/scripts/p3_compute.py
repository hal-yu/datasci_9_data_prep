import pandas as pd
import pickle
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Import the data
df = pd.read_csv('model_dev2/data/processed/mapping_live_births.csv')
len(df)

# Drop rows with missing values
df.dropna(inplace=True)
len(df)

# Define the features and the target variable "number_of_live_births"
X = df.drop('number_of_live_births', axis=1) 
y = df['number_of_live_births'] 

# Initialize the StandardScaler
scaler = StandardScaler()
scaler.fit(X)
pickle.dump(scaler, open('model_dev2/models/scaler_live_births.sav', 'wb')) #The "wb" mode ensures that the file is opened for writing in binary mode.

# Fit the scaler to the features and transform
X_scaled = scaler.transform(X)

# Split the scaled data into training, validation, and testing sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Check the size of each set
(X_train.shape, X_val.shape, X_test.shape)

# Pkle the X_train for later use in explanation
pickle.dump(X_train, open('model_dev2/models/X_train_live_births.sav', 'wb'))
# Pkle X.columns for later use in explanation
pickle.dump(X.columns, open('model_dev2/models/X_columns_live_births.sav', 'wb'))

# Import the data
df = pd.read_csv('model_dev2/data/processed/mapping_infant_mortality_rate.csv')
len(df)

# Drop rows with missing values
df.dropna(inplace=True)
len(df)

# Define the features and the target variable "infant_mortality_rate"
X = df.drop('infant_mortality_rate', axis=1) 
y = df['infant_mortality_rate'] 

# Initialize the StandardScaler
scaler = StandardScaler()
scaler.fit(X)
pickle.dump(scaler, open('model_dev2/models/scaler_infant_mortality_rate.sav', 'wb')) #The "wb" mode ensures that the file is opened for writing in binary mode.

# Fit the scaler to the features and transform
X_scaled = scaler.transform(X)

# Split the scaled data into training, validation, and testing sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Check the size of each set
(X_train.shape, X_val.shape, X_test.shape)

# Pkle the X_train for later use in explanation
pickle.dump(X_train, open('model_dev2/models/X_train_infant_mortality_rate.sav', 'wb'))
# Pkle X.columns for later use in explanation
pickle.dump(X.columns, open('model_dev2/models/X_columns_infant_mortality_rate.sav', 'wb'))