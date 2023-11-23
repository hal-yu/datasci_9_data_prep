import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

## get data raw
df = pd.read_pickle('model_dev1/data/raw/cause_of_death.pkl')

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
    'year',
    'leading_cause',
    'sex',
    'race_ethnicity',
    'deaths',
    'age_adjusted_death_rate'
]

df = df[to_keep]
df.dropna(inplace=True)

## Convert 'sex' to integers
df['sex'] = df['sex'].astype(int)

## Perform Ordinal encoding on sex
enc = OrdinalEncoder()
enc.fit(df[['sex']])
df['sex'] = enc.transform(df[['sex']])

## Create dataframe with mapping
df_mapping_sex = pd.DataFrame(enc.categories_[0], columns=['sex'])
df_mapping_sex['sex_ordinal'] = df_mapping_sex.index
df_mapping_sex
## save mapping to csv
df_mapping_sex.to_csv('model_dev1/data/processed/mapping_sex.csv', index=False)

## perform ordinal encoding on race_ethnicity
enc = OrdinalEncoder()
enc.fit(df[['race_ethnicity']])
df['race_ethnicity'] = enc.transform(df[['race_ethnicity']])

## Create dataframe with mapping
df_mapping_race_ethnicity = pd.DataFrame(enc.categories_[0], columns=['race_ethnicity'])
df_mapping_race_ethnicity['race_ethnicity_ordinal'] = df_mapping_race_ethnicity.index
df_mapping_race_ethnicity
## save mapping to csv
df_mapping_race_ethnicity.to_csv('model_dev1/data/processed/mapping_race_ethnicity.csv', index=False)

## perform ordinal encoding on leading_cause
enc = OrdinalEncoder()
enc.fit(df[['leading_cause']])
df['leading_cause'] = enc.transform(df[['leading_cause']])

## create dataframe with mapping
df_mapping_cause = pd.DataFrame(enc.categories_[0], columns=['leading_cause'])
df_mapping_cause['leading_cause_ordinal'] = df_mapping_cause.index
df_mapping_cause.head(5)
# save mapping to csv
df_mapping_cause.to_csv('model_dev1/data/processed/mapping_leading_cause.csv', index=False)