import pandas as pd
import numpy as np
import os
from pathlib import Path

# Raíz del proyecto: sube 3 niveles desde data/scripts/cifrado/
BASE_DIR = Path(__file__).resolve().parents[3]

path  = BASE_DIR / 'data' / 'processed' / 'Darknet.parquet'
path2 = BASE_DIR / 'data' / 'processed' / 'Encryption_Dataset_Clean.parquet'


df = pd.read_parquet(path2, engine='pyarrow')


print("\nDarknet:")
print(df.info())
print(df['Encryption_Label'].value_counts())
