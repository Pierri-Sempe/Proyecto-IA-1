import pandas as pd
import math
from naive_bayes import NaiveBayesManual

def k_folds_manual(textos, etiquetas, k=5):
    """K-Folds sobre el set de entrenamiento (80% del total)"""
    n = len(textos)
    tamaño_fold = n // k

    textos    = list(textos)
    etiquetas = list(etiquetas)

    resultados_por_fold = []

    for i in range(k):
        print(f"\n--- Fold {i+1}/{k} ---")

        inicio = i * tamaño_fold
        fin    = inicio + tamaño_fold

        textos_prueba    = textos[inicio:fin]
        etiquetas_prueba = etiquetas[inicio:fin]

        textos_entrenamiento    = textos[:inicio] + textos[fin:]
        etiquetas_entrenamiento = etiquetas[:inicio] + etiquetas[fin:]

        modelo = NaiveBayesManual()
        modelo.entrenar(textos_entrenamiento, etiquetas_entrenamiento)

        predicciones = [modelo.predecir(t) for t in textos_prueba]
        metricas = calcular_metricas(etiquetas_prueba, predicciones, modelo.clases)
        resultados_por_fold.append(metricas)

        print(f"Accuracy: {metricas['accuracy']:.4f}")

    # Promediar métricas
    print("\n=== RESULTADOS K-FOLDS (sobre set de entrenamiento) ===")
    accuracy_promedio = sum(r['accuracy'] for r in resultados_por_fold) / k
    print(f"Accuracy promedio: {accuracy_promedio:.4f}")

    return resultados_por_fold


def evaluar_test_final(modelo, textos_test, etiquetas_test):
    """Evaluación final sobre el 20% que nunca se usó"""
    print("\n=== EVALUACIÓN FINAL (sobre test set — 20%) ===")
    clases = list(set(etiquetas_test))
    predicciones = [modelo.predecir(t) for t in textos_test]
    metricas = calcular_metricas(etiquetas_test, predicciones, clases)
    print(f"Accuracy final: {metricas['accuracy']:.4f}")
    return metricas


def calcular_metricas(reales, predicciones, clases):
    """Calcula Precisión, Recall, F1 y Accuracy"""

    correctos = sum(r == p for r, p in zip(reales, predicciones))
    accuracy  = correctos / len(reales)

    metricas_por_clase = {}

    for clase in sorted(clases):
        tp = sum(1 for r, p in zip(reales, predicciones) if r == clase and p == clase)
        fp = sum(1 for r, p in zip(reales, predicciones) if r != clase and p == clase)
        fn = sum(1 for r, p in zip(reales, predicciones) if r == clase and p != clase)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metricas_por_clase[clase] = {
            'precision': precision, 'recall': recall, 'f1': f1
        }
        print(f"  {clase}: P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")

    # Macro F1
    macro_f1 = sum(m['f1'] for m in metricas_por_clase.values()) / len(clases)
    print(f"\n  Macro F1: {macro_f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")

    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'por_clase': metricas_por_clase}


if __name__ == '__main__':
    import pickle

    # Cargar sets ya divididos por entrenar.py
    df_train = pd.read_csv('data/train_set.csv')
    df_test  = pd.read_csv('data/test_set.csv')

    print(f"Set de entrenamiento: {len(df_train)} tickets")
    print(f"Set de test final:    {len(df_test)} tickets")

    textos_train    = df_train['instruction'].tolist()
    etiquetas_train = df_train['category'].tolist()

    textos_test    = df_test['instruction'].tolist()
    etiquetas_test = df_test['category'].tolist()

    # 1. K-Folds sobre el set de entrenamiento
    resultados_folds = k_folds_manual(textos_train, etiquetas_train, k=5)

    # 2. Evaluación final sobre el test set
    with open('modelo/modelo_entrenado.pkl', 'rb') as f:
        modelo = pickle.load(f)

    evaluar_test_final(modelo, textos_test, etiquetas_test)

    print("\n=== TABLA RESUMEN DE FOLDS ===")
    for i, r in enumerate(resultados_folds):
        macro = sum(m['f1'] for m in r['por_clase'].values()) / len(r['por_clase'])
        print(f"Fold {i+1}: Accuracy={r['accuracy']:.4f}  MacroF1={macro:.4f}")