import pickle
import pandas as pd

with open('modelo/modelo_entrenado.pkl', 'rb') as f:
    modelo = pickle.load(f)

df_test = pd.read_csv('data/test_set.csv')
textos_test = df_test['instruction'].tolist()
etiquetas_test = df_test['category'].tolist()

predicciones = [modelo.predecir(t) for t in textos_test]

clases = sorted(set(etiquetas_test))

print("Clases:", clases)
print("\nMatriz de Confusión:")
print("(filas = real, columnas = predicho)")
print()

header = f"{'':15}" + "".join(f"{c:15}" for c in clases)
print(header)

for clase_real in clases:
    fila = f"{clase_real:15}"
    for clase_pred in clases:
        count = sum(1 for r, p in zip(etiquetas_test, predicciones) 
                   if r == clase_real and p == clase_pred)
        fila += f"{count:15}"
    print(fila)