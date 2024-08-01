import pandas as pd

INDEX_PRESETS_FILEPATH = r'./data/IndexPresets.xlsx'

def load_index_presets(mode):

    df = pd.read_excel(INDEX_PRESETS_FILEPATH, sheet_name="Data")

    if mode == 'dropdown':
        data = df[['label', 'value']].to_dict(orient='records')
    elif mode == 'data':
        df['combined'] = df.apply(lambda row: [row['average return'], row['standard deviation']], axis=1)
        data = df[['value','combined']].set_index('value')['combined'].to_dict()

    return data