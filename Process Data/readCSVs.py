import pandas as pd
import numpy as np


def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    output = '.'.join([i, (d + '0' * n)[:n]])
    return float(output)


def readVoltage(data):
    df = pd.read_csv(data, comment='#', header=1)
    df.drop_duplicates(subset='Time (s)', keep='last', inplace=True)
    df.dropna(inplace=True)
    df.drop(df.columns.difference(['C1 True RMS (Ṽ)', 'C2 True RMS (Ṽ)', 'Time (s)']), 1, inplace=True)
    df = df.rename(columns={'C1 True RMS (Ṽ)': 'C1 True RMS', 'C2 True RMS (Ṽ)': 'C2 True RMS',
                            'Time (s)': 'Time V'})
    return df


def fillTimeHoles(data):
    for i in data.index:
        if i == 0:
            continue
        previous = int(data['Time'][i - 1] * 10)
        current = int(data['Time'][i] * 10)
        gap = (current - previous) * .1
        if gap > .11:
            fill = []
            for j in range(previous + 1, current, 1):
                fill.append([truncate(j * .1, 1), 0, 0])
            fill = pd.DataFrame(fill, columns=['Time', 'Strain', 'Position'])
            data = pd.concat([data.iloc[:i - 1], fill, data.iloc[i:]]).reset_index(drop=True)
    return data.reset_index(drop=True)


def correctTimes(data):
    for i in data.index:
        data['Time'][i] = truncate(i * .1, 1)
    return data.reset_index(drop=True)


def positionToStrain(data):
    data['Strain'] = np.zeros(len(data))
    originalLength = 2
    for index, row in data.iterrows():
        length = row['Position']
        row['Strain'] = (length - originalLength) / originalLength


def readStrain(data, howToFill):
    df = pd.read_csv(data)
    df.drop_duplicates(subset='Total Time (s)', keep='last', inplace=True)
    df.drop(df.columns.difference(['Total Time (s)', 'Position(Linear:Position) (in)']), 1, inplace=True)
    df = df.rename(columns={'Total Time (s)': 'Time S', 'Position(Linear:Position) (in)': 'Position'})
    positionToStrain(df)
    df.reset_index(inplace=True, drop=True)
    if howToFill == 1:
        df = fillTimeHoles(df)
    elif howToFill == 2:
        df = correctTimes(df)
    strain = df.loc[:, 'Strain'].to_numpy()
    strain = np.flip(strain)
    df['Strain'] = strain
    return df
