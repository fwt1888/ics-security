import pandas as pd


for chunk in pd.read_csv('G:\dataset\ics\wustl-iiot\wustl_iiot_2021.csv', chunksize=1):

    if chunk.iloc[0]['Traffic'] != 'normal':
        print(chunk)

    if chunk.iloc[0]['sDSb'] != 0:
        print(chunk)

    print('SAppBytes',chunk.iloc[0]['SAppBytes'])
    print('Mean',chunk.iloc[0]['Mean'])
    print('sDSb', chunk.iloc[0]['sDSb'])
    print('Traffic',chunk.iloc[0]['Traffic'])
