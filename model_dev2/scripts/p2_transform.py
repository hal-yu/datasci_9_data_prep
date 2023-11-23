import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

## get data raw
df = pd.read_pickle('model_dev2/data/raw/infant_mortality.pkl')

## get column names
df.columns

## Clean column names (make all lower case and remove white space)
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns

## Data Types
df.dtypes # nice combination of numbers and strings/objects 
len(df)

# keep columns 
to_keep = [
    'materal_race_or_ethnicity',
    'infant_mortality_rate',
    'number_of_live_births',
    'infant_deaths',
]

df = df[to_keep]
df.dropna(inplace=True)

## Perform Ordinal encoding on sex
enc = OrdinalEncoder()
enc.fit(df[['materal_race_or_ethnicity']])
df['materal_race_or_ethnicity'] = enc.transform(df[['materal_race_or_ethnicity']])

## Create dataframe with mapping
df_mapping_race = pd.DataFrame(enc.categories_[0], columns=['materal_race_or_ethnicity'])
df_mapping_race['race_ordinal'] = df_mapping_race.index
df_mapping_race
## save mapping to csv
df_mapping_race.to_csv('model_dev2/data/processed/mapping_race.csv', index=False)

## perform ordinal encoding on infant mortality rate
enc = OrdinalEncoder()
enc.fit(df[['infant_mortality_rate']])
df['infant_mortality_rate'] = enc.transform(df[['infant_mortality_rate']])

## Create dataframe with mapping
df_mapping_infant_mortality_rate = pd.DataFrame(enc.categories_[0], columns=['infant_mortality_rate'])
df_mapping_infant_mortality_rate['infant_mortality_rate_ordinal'] = df_mapping_infant_mortality_rate.index
df_mapping_infant_mortality_rate
## save mapping to csv
df_mapping_infant_mortality_rate.to_csv('model_dev2/data/processed/mapping_infant_mortality_rate.csv', index=False)

## perform ordinal encoding on live births
enc = OrdinalEncoder()
enc.fit(df[['number_of_live_births']])
df['number_of_live_births'] = enc.transform(df[['number_of_live_births']])

## Create dataframe with mapping
df_mapping_live_births = pd.DataFrame(enc.categories_[0], columns=['number_of_live_births'])
df_mapping_live_births['live_births_ordinal'] = df_mapping_live_births.index
df_mapping_live_births.head(5)
# save mapping to csv
df_mapping_live_births.to_csv('model_dev2/data/processed/mapping_live_births.csv', index=False)