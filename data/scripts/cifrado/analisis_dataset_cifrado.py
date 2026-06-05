import pandas as pd
import numpy as np
import os

path = r'C:\Users\Felix\Desktop\Tesis\data\processed\Darknet.parquet'
path2 = r'C:\Users\i21327\Desktop\Tesis\Modelo_Hibrido_Ligero_Tesis\data\processed\Encryption_Dataset_Clean.parquet'


df = pd.read_parquet(path2, engine='pyarrow')


print("\nDarknet:")
print(df.info())
print(df['Encryption_Label'].value_counts())
