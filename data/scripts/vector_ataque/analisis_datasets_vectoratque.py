import pandas as pd
import numpy as np
import os

path = r'C:\Users\Felix\Desktop\Tesis\data\processed\CICIDS2017.parquet'
path2 = r'C:\Users\Felix\Desktop\Tesis\data\processed\UNSW-NB15-V3.parquet'
path3 = r'C:\Users\Felix\Desktop\Tesis\data\processed\Ataques_Unificado_Raw.parquet'


df = pd.read_parquet(path)
df2 = pd.read_parquet(path2)
df3 = pd.read_parquet(path3)


print("\nCICIDS2017:")
print(df.info())
print("\nUNSW-NB15-V3:")
print(df2.info())
print("\nAtaques_Unificado_Raw:")
print(df3['attack_vector_raw'].value_counts())





