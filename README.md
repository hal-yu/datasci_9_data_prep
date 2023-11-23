# datasci_9_data_prep
Focus on selecting datasets suitable for a machine learning experiment, with an emphasis on data cleaning, encoding, and transformation steps necessary to prepare the data. 

# Dataset 1 - Leading Cause of Death in NYC
```
Data retrieved from: https://data.cityofnewyork.us/Health/New-York-City-Leading-Causes-of-Death/jb7j-dtam
```
This dataset contains an overview of the leading causes of death in New York City since 2007. The causes of death is derived from the NYC death certificate issued for every death in NYC. This dataset provides information about the year, leading cause, sex, race/ethnicity, deaths, death rate, and the age adjusted death rate.


# Dataset 2 - Infant Mortality
```
Data retrieved from: https://data.cityofnewyork.us/Health/Infant-Mortality/fcau-jc6k
```
This dataset contains an overview of the infant mortality rates for New York City from 2007 to 2016. Infant death is counted when they are under 1 years of age and based off the death certificates. This datset provides information about the year, maternal race/ethnicity, infant mortality rate and infant deaths, neonatal mortality rate and infant deaths, postneonatal mortality rate and infant deaths, and the number of live births.

# Cleaning and transforming data
- Data was cleaned by changing column names to lower case and replacing white space with "_".
- Filter data by keeping important columns
- Ordinally encode columns 
```
For dataset 1, these columns were ordinally encoded and saved into the processed data folder for dataset1
- sex
- race/ethnicity
- leading cause of death
```

```
For dataset 2, these columns were ordinally encoded and saved into the processed data folder for dataset2
- infant mortality rate
- live birth rate
- race/ethnicity of mother
```

# Data Splitting
- After loading dataset, define the targetted variable and initialize the standard scaler
- Data was split dataset into training data, validation data, and testing data (70%, 15%, 15%)
- Use pickle to save the two variables (X_train and X_columns) into the models folder

