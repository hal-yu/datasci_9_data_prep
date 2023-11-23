import pandas as pd 

# original link: https://data.cityofnewyork.us/Health/New-York-City-Leading-Causes-of-Death/jb7j-dtam 
# data download link: 
datalink = 'https://data.cityofnewyork.us/resource/jb7j-dtam.csv'

df = pd.read_csv(datalink)
df
df.size
df.sample(5)

## save as csv
df.to_csv('model_dev1/data/raw/cause_of_death.csv', index=False)

## save as pickle
df.to_pickle('model_dev1/data/raw/cause_of_death.pkl')
 
