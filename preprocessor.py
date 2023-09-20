import pandas as pd

def split_cabin(x):
    if pd.isna(x):
        return 'Missing'
    else:
        return x.split('/')

def preprocessing(df):
    df['HomePlanet'].fillna('Missing', inplace=True)
    df['CryoSleep'].fillna('Missing', inplace=True)
    # Cabin preprocessing, extract deck / num / side
    df['Deck'] = df['Cabin'].apply(lambda x: split_cabin(x)[0])
    # df['Num'] = df['Cabin'].apply(lambda x: split_cabin(x)[1])
    df['Side'] = df['Cabin'].apply(lambda x: split_cabin(x)[2])
    df.drop('Cabin',axis=1, inplace=True)
    df['Destination'].fillna('Missing', inplace=True)
    df['Age'].fillna(df['Age'].mean(), inplace=True) # (mean)
    
    df['RoomService'].fillna(0, inplace=True)
    df['FoodCourt'].fillna(0, inplace=True)
    df['ShoppingMall'].fillna(0, inplace=True)
    df['Spa'].fillna(0, inplace=True)
    df['VRDeck'].fillna(0, inplace=True)
    
    df['VIP'].fillna('Missing', inplace=True)
    df.drop('Name', axis=1, inplace=True)
    return df