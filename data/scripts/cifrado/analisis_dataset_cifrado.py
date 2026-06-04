import pandas as pd
import numpy as np
import os

path = r'C:\Users\Felix\Desktop\Tesis\data\processed\Darknet.parquet'


df = pd.read_parquet(path)


print("\nDarknet:")
print(df.info())
print(df['Label'].value_counts())