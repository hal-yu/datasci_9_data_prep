import pandas as pd 

# original link: https://data.cityofnewyork.us/Health/Infant-Mortality/fcau-jc6k 
# data download link: 
datalink = 'https://data.cityofnewyork.us/resource/fcau-jc6k.csv'

df = pd.read_csv(datalink)
df
df.size
df.sample(5)

## save as csv
df.to_csv('model_dev2/data/raw/infant_mortality.csv', index=False)

## save as pickle
df.to_pickle('model_dev2/data/raw/infant_mortality.pkl')
 
