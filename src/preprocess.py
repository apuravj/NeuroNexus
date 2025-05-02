# preprocess

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(path):
    df = pd.read_csv()      ##  credit card data --> https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    
    # Normalize 'Amount'
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    
    # Drop 'Time'
    df.drop(['Time'], axis=1, inplace=True)
    
    return df
