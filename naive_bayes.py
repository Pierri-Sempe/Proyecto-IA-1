import math
from collections import defaultdict
from preprocesamiento import preprocesar_texto

class NaiveBayesManual:
    
    def __init__(self):
        self.vocabulario = set()
        self.conteo_palabras_por_clase = {}   # {clase: {palabra: conteo}}
        self.conteo_total_palabras = {}        # {clase: total de palabras}
        self.conteo_docs_por_clase = {}        # {clase: número de tickets}
        self.total_docs = 0
        self.clases = []
    
    def entrenar(self, textos, etiquetas):
        """
        textos: lista de tickets (strings)
        etiquetas: lista de categorías correspondientes
        """
        self.total_docs = len(textos)
        self.clases = list(set(etiquetas))
        
        # Inicializar contadores
        for clase in self.clases:
            self.conteo_palabras_por_clase[clase] = defaultdict(int)
            self.conteo_total_palabras[clase] = 0
            self.conteo_docs_por_clase[clase] = 0
        
        # Contar palabras por clase y construir vocabulario
        for texto, etiqueta in zip(textos, etiquetas):
            palabras = preprocesar_texto(texto)
            self.conteo_docs_por_clase[etiqueta] += 1
            
            for palabra in palabras:
                self.vocabulario.add(palabra)
                self.conteo_palabras_por_clase[etiqueta][palabra] += 1
                self.conteo_total_palabras[etiqueta] += 1
    
    def predecir(self, texto):
        """Retorna la clase con mayor puntaje"""
        palabras = preprocesar_texto(texto)
        puntajes = {}
        V = len(self.vocabulario)  # tamaño del vocabulario (para Laplace)
        
        for clase in self.clases:
            # Probabilidad a priori en logaritmo
            prob_clase = math.log(
                self.conteo_docs_por_clase[clase] / self.total_docs
            )
            
            # Sumar log(P(palabra | clase)) para cada palabra
            suma_log = 0
            for palabra in palabras:
                conteo = self.conteo_palabras_por_clase[clase].get(palabra, 0)
                # Laplace Smoothing: suma 1 al numerador, V al denominador
                prob_palabra = (conteo + 1) / (self.conteo_total_palabras[clase] + V)
                suma_log += math.log(prob_palabra)
            
            puntajes[clase] = prob_clase + suma_log
        
        # Retornar la clase con el puntaje más alto
        return max(puntajes, key=puntajes.get)
    
    def predecir_con_probabilidades(self, texto):
        """Retorna todos los puntajes (útil para la web)"""
        palabras = preprocesar_texto(texto)
        puntajes = {}
        V = len(self.vocabulario)
        
        for clase in self.clases:
            prob_clase = math.log(
                self.conteo_docs_por_clase[clase] / self.total_docs
            )
            suma_log = 0
            for palabra in palabras:
                conteo = self.conteo_palabras_por_clase[clase].get(palabra, 0)
                prob_palabra = (conteo + 1) / (self.conteo_total_palabras[clase] + V)
                suma_log += math.log(prob_palabra)
            puntajes[clase] = prob_clase + suma_log
        
        return puntajes