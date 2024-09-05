import pandas as pd


for chunk in pd.read_csv('G:\dataset\ics\wustl-iiot\wustl_iiot_2021.csv', chunksize=1):

    print(chunk)
