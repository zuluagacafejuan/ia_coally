import nltk
from unidecode import unidecode
import re
from nltk.corpus import stopwords
from googletrans import Translator
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import numpy as np

traductor = Translator()
stemmer = SnowballStemmer('spanish')
stopwords_espanol = set(stopwords.words('spanish'))
lemmatizer = WordNetLemmatizer()

#Funcion de transformacion para el pipeline
def procesar_texto_docs(corpus):
  def procesar_texto(texto):
    # Traducir texto a español si está en inglés

    patron_url = re.compile(r"http[s]?://\S+")
    texto_sin_urls = patron_url.sub('', texto)
    texto_sin_urls = texto_sin_urls.replace('_', ' ')
    try:
      resultado_traduccion = traductor.translate(texto_sin_urls, dest='es').text
    except:
      resultado_traduccion = texto_sin_urls

    resultado_sin_acentos = unidecode(resultado_traduccion)

    # Eliminar puntuación del texto traducido
    resultado_sin_puntuacion = re.sub(r'[\W\d]+', ' ', resultado_sin_acentos)

    # Eliminar stopwords del texto traducido

    palabras = nltk.word_tokenize(resultado_sin_puntuacion.lower())
    palabras_filtradas = [palabra for palabra in palabras if palabra not in stopwords_espanol]

    # Stemming del texto filtrado

    palabras_stemmed = [stemmer.stem(palabra) for palabra in palabras_filtradas]

    # Lemmatization del texto filtrado

    palabras_lemmatized = [lemmatizer.lemmatize(palabra) for palabra in palabras_stemmed]

    return ' '.join(palabras_lemmatized)

  if isinstance(corpus, str):
    return
  else:
    return [procesar_texto(texto) for texto in corpus]
  

class ensamble_model():

  def __init__(self, scaler):
    self.models = {}

  def agregar_modelo(self, name, model):
    self.models[name] = model

  def quitar_modelo(self, name):
    self.models.pop(name)

  def predict_proba(self, X):
    results = {}
    for name, model in self.models.items():
      if name != 'SVC':
        results[name] = [i[1] for i in model.predict_proba(X)]
      if name == 'SVC':
        results[name] = [i for i in model.predict(X)]

    averages = [sum(values) / len(values) for values in zip(*results.values())]
    return np.array(averages)