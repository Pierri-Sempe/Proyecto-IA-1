import pandas as pd

# Cargar el dataset
df = pd.read_csv('data/customer_support_tickets.csv')

# Ver las primeras filas
print(df.head())

# Ver qué columnas tiene
print(df.columns)

# Ver cuántos tickets hay por categoría
print(df['Ticket Type'].value_counts())  # el nombre de la columna puede variar