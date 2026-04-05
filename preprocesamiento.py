import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Cargar herramientas de NLTK
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocesar_texto(texto):
    # 1. Convertir a minúsculas
    texto = texto.lower()
    
    # 2. Eliminar números y caracteres especiales (solo dejar letras y espacios)
    texto = re.sub(r'[^a-z\s]', '', texto)
    
    # 3. Separar el texto en palabras (tokenizar)
    palabras = texto.split()
    
    # 4. Eliminar stopwords (palabras sin significado: "the", "is", "a"...)
    palabras = [p for p in palabras if p not in stop_words]
    
    # 5. Lematizar (reducir palabras a su raíz: "running" → "run")
    palabras = [lemmatizer.lemmatize(p) for p in palabras]
    
    return palabras  # retorna una lista de palabras limpias

# Prueba rápida
if __name__ == '__main__':
    ejemplo = "My internet connection is not working properly!!!"
    print(preprocesar_texto(ejemplo))
    # Resultado esperado: ['internet', 'connection', 'work', 'proper']