# Proyecto — Clasificación Automática de Tickets

Sistema de clasificación automática de tickets de soporte al cliente usando el algoritmo 
Naïve Bayes Multinomial, implementado desde cero en Python e integrado con una interfaz web funcional.

Desarrollado para el curso de Inteligencia Artificial — Universidad Rafael Landívar, Primer Semestre 2026.

---

## ¿Qué hace el sistema?

El usuario ingresa una solicitud de soporte (asunto y descripción) y el sistema la clasifica 
automáticamente en una de las siguientes 11 categorías:

| Categoría | Descripción |
|---|---|
| ACCOUNT | Problemas de cuenta y acceso |
| ORDER | Gestión de pedidos |
| REFUND | Devoluciones |
| PAYMENT | Pagos y cobros |
| DELIVERY | Entregas |
| SHIPPING | Dirección de envío |
| INVOICE | Facturas |
| CONTACT | Contacto y soporte |
| FEEDBACK | Opiniones y quejas |
| CANCEL | Cancelaciones |
| SUBSCRIPTION | Suscripciones |

---

## Tecnologías utilizadas

- **Python 3.10+**
- **Flask** — servidor web
- **NLTK** — tokenización y stopwords
- **Pandas** — manejo del dataset
- **HTML / CSS / JavaScript** — interfaz web

---

## Estructura del proyecto


proyecto/
│
├── data/
│   ├── Dataset.csv              ← Dataset original Bitext (26,872 tickets)
│   ├── train_set.csv            ← 80% para entrenamiento (generado automáticamente)
│   └── test_set.csv             ← 20% para prueba final (generado automáticamente)
│
├── modelo/
│   └── modelo_entrenado.pkl     ← Modelo Naïve Bayes entrenado y guardado
│
├── web/
│   └── templates/
│       └── index.html           ← Interfaz web del portal de soporte
│
├── preprocesamiento.py          ← Limpieza, tokenización, stopwords y lematización
├── naive_bayes.py               ← Algoritmo Naïve Bayes implementado manualmente
├── entrenar.py                  ← Entrena el modelo y guarda el archivo .pkl
├── evaluar.py                   ← K-Folds Cross Validation y métricas
├── app.py                       ← Servidor Flask — integración backend con frontend
└── README.md  

---

## Instalación

### 1. Clona el repositorio

```bash
git clone https://github.com/tu-usuario/tu-repositorio.git
cd tu-repositorio
```

### 2. Instala las dependencias

```bash
pip install flask nltk pandas
```

### 3. Descarga los recursos de NLTK

Ejecuta esto una sola vez en tu terminal:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 4. Descarga el dataset

El dataset utilizado es el **Bitext Gen AI Chatbot Customer Support Dataset**, disponible en:

https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset

Descárgalo, renómbralo como `Dataset.csv` y colócalo dentro de la carpeta `data/`.

---

## Ejecución

Sigue este orden la primera vez:

### Paso 1 — Entrenar el modelo
```bash
python entrenar.py
```
Esto divide el dataset en 80% entrenamiento y 20% test, entrena el modelo y guarda 
el archivo `modelo/modelo_entrenado.pkl`.

### Paso 2 — Evaluar el modelo (opcional)
```bash
python evaluar.py
```
Ejecuta K-Folds Cross Validation con K=5 y muestra las métricas por clase 
(Precisión, Recall, F1-Score, Accuracy y Macro F1).

### Paso 3 — Iniciar la aplicación web
```bash
python app.py
```
Abre tu navegador en:
```
http://127.0.0.1:5000
```

---

## Resultados del modelo

| Métrica | Valor |
|---|---|
| Accuracy promedio K-Folds | 99.66% |
| Accuracy test final | 99.81% |
| Macro F1 test final | 99.80% |
| Tickets evaluados | 5,375 |
| Errores cometidos | 10 |

---

## Algoritmo implementado

- **Bag of Words** — construcción del vocabulario desde el corpus de entrenamiento
- **Naïve Bayes Multinomial** — implementado manualmente sin scikit-learn ni equivalentes
- **Laplace Smoothing** — para evitar probabilidades cero en palabras no vistas
- **Suma de logaritmos** — para evitar underflow numérico durante la inferencia
- **K-Folds Cross Validation (K=5)** — implementado manualmente para evaluación rigurosa

---

## Autor

- Hans Pierr Sempé Aquino - carné 1083920

Universidad Rafael Landívar — Facultad de Ingeniería  
Inteligencia Artificial, Primer Semestre 2026
