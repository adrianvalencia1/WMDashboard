from datetime import datetime, timedelta
import pandas as pd

filepath = r"C:\Users\Adrian(SatovskyAsset\Desktop\Projects\IFA\SeriesReport-20240610103427_33cb36.xlsx"

def read_inflation_data(filepath):
    df = pd.read_excel(filepath)
    title = df.columns[0]
    df = df.iloc[11:]
    df.drop(title, axis=1, inplace=True)
    df['Date'] = None

    for index, row in df.iterrows():
        if row['Unnamed: 2'][0] != 'S':
            date = str(row['Unnamed: 1']) + '-' + row['Unnamed: 2'][1:]
            df.at[index,'Date'] = date


    df.dropna(inplace=True)
    df.set_index('Date', inplace=True)
    df.drop(['Unnamed: 1', 'Unnamed: 2'], axis=1, inplace=True)
    df.rename(columns={'Unnamed: 3': 'CPI'}, inplace=True)

    # adjust for future inflation
    today = datetime.today()

    while today > pd.to_datetime(df.index[-1]):
        next_month = datetime.strptime(df.index[-1], '%Y-%m') + timedelta(days=31)
        next_month = next_month.strftime('%Y-%m')
        new_row = pd.DataFrame({'CPI':df['CPI'].iloc[-1]}, index=[next_month])
        df = pd.concat([df,new_row])

    return df

def calculate_inflation(df, initial_amount):
    # new - old / old + $amount = new worth of $
    df['Inflation Rate'] = df.pct_change()
    df['Buying Power'] = initial_amount * (1 + df['Inflation Rate']).cumprod()
    df.dropna(inplace=True)
    return df

def append_inflation(inf_df, datasets):
    for key, df in datasets.items():
        df = df.join(inf_df['Inflation Rate'])