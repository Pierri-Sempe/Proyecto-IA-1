import pandas as pd
import pickle
from naive_bayes import NaiveBayesManual

# Cargar dataset
df = pd.read_csv('data/customer_support_tickets.csv')

COLUMNA_TEXTO = 'Ticket Description'   
COLUMNA_ETIQUETA = 'Ticket Type'       

# Eliminar filas con valores nulos
df = df.dropna(subset=['Ticket Description', 'Ticket Type'])

textos = df[COLUMNA_TEXTO].tolist()
etiquetas = df[COLUMNA_ETIQUETA].tolist()

# Entrenar el modelo
modelo = NaiveBayesManual()
modelo.entrenar(textos, etiquetas)

print("Modelo entrenado con", len(textos), "tickets")
print("Categorías:", modelo.clases)

# Guardar el modelo en un archivo
with open('modelo/modelo_entrenado.pkl', 'wb') as f:
    pickle.dump(modelo, f)

print("Modelo guardado en modelo/modelo_entrenado.pkl")