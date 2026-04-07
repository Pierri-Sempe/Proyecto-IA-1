import pandas as pd

# Cargar el dataset
df = pd.read_csv('data/Dataset.csv')

# Ver las primeras filas
print(df.head())

# Ver qué columnas tiene
print(df.columns)

# Ver cuántos tickets hay por categoría
print(df['category'].value_counts())  