import pandas as pd
import pickle
import random
from naive_bayes import NaiveBayesManual

# Cargar dataset
df = pd.read_csv('data/Dataset.csv')
df = df.dropna(subset=['instruction', 'category'])

# Mezclar los datos aleatoriamente antes de dividir
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# División 80% entrenamiento / 20% test final
corte = int(len(df) * 0.80)
df_entrenamiento = df[:corte]
df_test          = df[corte:]

print(f"Total de tickets:      {len(df)}")
print(f"Tickets entrenamiento: {len(df_entrenamiento)}")
print(f"Tickets test final:    {len(df_test)}")

# Guardar el set de test para usarlo después en evaluar.py
df_test.to_csv('data/test_set.csv', index=False)
df_entrenamiento.to_csv('data/train_set.csv', index=False)
print("Sets guardados en data/train_set.csv y data/test_set.csv")

# Entrenar el modelo con el 80%
textos    = df_entrenamiento['instruction'].tolist()
etiquetas = df_entrenamiento['category'].tolist()

modelo = NaiveBayesManual()
modelo.entrenar(textos, etiquetas)

print("\nModelo entrenado con", len(textos), "tickets")
print("Categorías:", modelo.clases)

# Guardar el modelo
with open('modelo/modelo_entrenado.pkl', 'wb') as f:
    pickle.dump(modelo, f)

print("Modelo guardado en modelo/modelo_entrenado.pkl")