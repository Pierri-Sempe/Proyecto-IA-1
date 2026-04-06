import pandas as pd
import math
from naive_bayes import NaiveBayesManual
from collections import defaultdict

def k_folds_manual(textos, etiquetas, k=5):
    """Divide los datos en K partes y evalúa K veces"""
    n = len(textos)
    tamaño_fold = n // k
    
    # Convertir a listas para poder hacer slices
    textos = list(textos)
    etiquetas = list(etiquetas)
    
    resultados_por_fold = []
    
    for i in range(k):
        print(f"\n--- Fold {i+1}/{k} ---")
        
        # Definir inicio y fin del fold de prueba
        inicio = i * tamaño_fold
        fin = inicio + tamaño_fold
        
        # Separar datos de prueba y entrenamiento
        textos_prueba = textos[inicio:fin]
        etiquetas_prueba = etiquetas[inicio:fin]
        
        textos_entrenamiento = textos[:inicio] + textos[fin:]
        etiquetas_entrenamiento = etiquetas[:inicio] + etiquetas[fin:]
        
        # Entrenar
        modelo = NaiveBayesManual()
        modelo.entrenar(textos_entrenamiento, etiquetas_entrenamiento)
        
        # Evaluar
        predicciones = [modelo.predecir(t) for t in textos_prueba]
        metricas = calcular_metricas(etiquetas_prueba, predicciones, modelo.clases)
        resultados_por_fold.append(metricas)
        
        print(f"Accuracy: {metricas['accuracy']:.4f}")
    
    # Promediar métricas de todos los folds
    print("\n=== RESULTADOS PROMEDIO ===")
    accuracy_promedio = sum(r['accuracy'] for r in resultados_por_fold) / k
    print(f"Accuracy promedio: {accuracy_promedio:.4f}")
    
    return resultados_por_fold

def calcular_metricas(reales, predicciones, clases):
    """Calcula Precisión, Recall, F1 y Accuracy"""
    
    # Contar correctos para accuracy
    correctos = sum(r == p for r, p in zip(reales, predicciones))
    accuracy = correctos / len(reales)
    
    metricas_por_clase = {}
    
    for clase in clases:
        # TP: predijo clase y era correcto
        tp = sum(1 for r, p in zip(reales, predicciones) if r == clase and p == clase)
        # FP: predijo clase pero era otra
        fp = sum(1 for r, p in zip(reales, predicciones) if r != clase and p == clase)
        # FN: era clase pero predijo otra
        fn = sum(1 for r, p in zip(reales, predicciones) if r == clase and p != clase)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metricas_por_clase[clase] = {
            'precision': precision, 'recall': recall, 'f1': f1
        }
        print(f"  {clase}: P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")
    
    return {'accuracy': accuracy, 'por_clase': metricas_por_clase}

# Ejecutar
if __name__ == '__main__':
    df = pd.read_csv('data/customer_support_tickets.csv')
    df = df.dropna(subset=['Ticket Description', 'Ticket Type'])
    
    textos = df['Ticket Description'].tolist()
    etiquetas = df['Ticket Type'].tolist()
    
    k_folds_manual(textos, etiquetas, k=5)