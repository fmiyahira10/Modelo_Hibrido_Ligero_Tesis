import pandas as pd
import glob
import os

path = os.path.join(os.path.dirname(__file__), '../../data/raw/CICIDS2017')
all_files = glob.glob(os.path.join(path, "*.csv"))

li=[]

for filename in all_files:
    df=pd.read_csv(filename, index_col=None, header=0, encoding='cp1252')
    df.columns = df.columns.str.strip()

    li.append(df)

df_total=pd.concat(li, axis=0, ignore_index=True)

output_path = os.path.join(os.path.dirname(__file__), '../../data/processed/CICIDS2017_unido.parquet')
df_total.to_parquet(output_path, index=False)
print("Archivo unificado creado exitosamente.")