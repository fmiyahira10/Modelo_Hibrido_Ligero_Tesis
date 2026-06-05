import pandas as pd
import numpy as np
import os
from pathlib import Path

# Raíz del proyecto: sube 3 niveles desde data/scripts/vector_ataque/
BASE_DIR = Path(__file__).resolve().parents[3]

#path  = BASE_DIR / 'data' / 'processed' / 'CICIDS2017.parquet'
#path2 = BASE_DIR / 'data' / 'processed' / 'UNSW-NB15-V3.parquet'
#path3 = BASE_DIR / 'data' / 'processed' / 'Ataques_Unificado_Raw.parquet'
path4 = BASE_DIR / 'data' / 'processed' / 'Attack_Dataset_Clean.parquet'


#df = pd.read_parquet(path)
df3 = pd.read_parquet(path4)


#print("\nCICIDS2017:")
#print(df.info())
#print("\nUNSW-NB15-V3:")
#print(df2.info())
#print("\nAtaques_Unificado_Raw:")
print(df3.info())
print(df3['attack_vector'].value_counts())





