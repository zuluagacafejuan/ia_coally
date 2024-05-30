import nltk
import time
nltk.download('stopwords')
nltk.download('punkt')
from fastapi import FastAPI
import requests
from pymongo import MongoClient
from googletrans import Translator
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import psycopg2
from nltk.corpus import stopwords
import pickle as pkl
import pandas as pd
import nltk
from unidecode import unidecode
import re
from bson.objectid import ObjectId
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Union



class CreateCVRequestModel(BaseModel):
  id_cv: str
  uniandes: Union[bool, None] = None

class CreateProjectRequestModel(BaseModel):
  id_project: str
  uniandes: Union[bool, None] = None

class AddApplicantRequestModel(BaseModel):
  id_cv: str
  id_project: str
  uniandes: Union[bool, None] = None

nltk.download('wordnet')

#########################################################
#################### FUNCIONES ÚTILES ###################
#########################################################

client = MongoClient("mongodb+srv://danielCTO:Coally2023-123@coally.nqokc.mongodb.net/CoallyProd?authSource=admin&replicaSet=atlas-39r1if-shard-0&w=majority&readPreference=primary&retryWrites=true&ssl=true")

db = client['CoallyProd']
db_proyectos = db['projects']
db_usuarios = db['users']
db_cvs = db['usercvs']

traductor = Translator()
stemmer = SnowballStemmer('spanish')
stopwords_espanol = set(stopwords.words('spanish'))
lemmatizer = WordNetLemmatizer()


r = requests.get('https://resumescreening-ml-coally.s3.amazonaws.com/carreras_a_majors_procesado.pkl')
carreras_a_majors_procesado = pkl.loads(r.content)
r = requests.get('https://resumescreening-ml-coally.s3.amazonaws.com/soft_skills_procesado.pkl')
soft_skills_procesado = pkl.loads(r.content)
r = requests.get('https://resumescreening-ml-coally.s3.amazonaws.com/technical_skills_procesado.pkl')
hard_skills_procesado = pkl.loads(r.content)
r = requests.get('https://resumescreening-ml-coally.s3.amazonaws.com/pipeline_clustering.pkl')

import os, tempfile
if r.status_code == 200:
    # Crear un archivo temporal para guardar el contenido descargado
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(r.content)
        temp_file_path = temp_file.name

    # Cargar el archivo desde el archivo temporal con joblib.load()
    pipeline = joblib.load(temp_file_path)

    # Eliminar el archivo temporal después de cargarlo
    os.unlink(temp_file_path)

    print("Archivo cargado exitosamente.")
else:
    print("Error al obtener el archivo desde la URL.")

r = requests.get('https://resumescreening-ml-coally.s3.amazonaws.com/model.pkl')
if r.status_code == 200:
    # Crear un archivo temporal para guardar el contenido descargado
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(r.content)
        temp_file_path = temp_file.name

    # Cargar el archivo desde el archivo temporal con joblib.load()
    modelo = joblib.load(temp_file_path)

    # Eliminar el archivo temporal después de cargarlo
    os.unlink(temp_file_path)

    print("Archivo cargado exitosamente.")
else:
    print("Error al obtener el archivo desde la URL.")

r = requests.get('https://resumescreening-ml-coally.s3.amazonaws.com/scaler.pkl')
scaler = pkl.loads(r.content)

r = requests.get('https://resumescreening-ml-coally.s3.amazonaws.com/keywords_procesado.pkl')
keywords_procesado = pkl.loads(r.content)


################################################################################################
################################ ACTUALIZAR COMPATIBILIDAD #####################################
################################################################################################
def conectar_base_datos():
    try:
        connection = psycopg2.connect(
            user="postgres",
            password="CoallySecur3",
            host="db-resumescreening-coally.c960wcwwcazt.us-east-2.rds.amazonaws.com",
            port="5432",
            database="postgres"
        )
        return connection
    except psycopg2.Error as error:
        print("Error al conectar a la base de datos:", error)
        return None


def actualizar_compatibilidad(connection, cursor, compatibilidad, id_cv, id_job, uniandes):
  try:
      if not uniandes:
        cursor.execute(f"""INSERT INTO public.general_compatibility (id_resume, id_job, compatibility) VALUES ('{id_cv}','{id_job}',{compatibilidad}) """.format(id_job=id_job, compatibilidad=compatibilidad, id_cv=id_cv))
        connection.commit()
      else:
        cursor.execute(f"""INSERT INTO public.general_compatibility_uniandes (id_resume, id_job, compatibility) VALUES ('{id_cv}','{id_job}',{compatibilidad}) """.format(id_job=id_job, compatibilidad=compatibilidad, id_cv=id_cv))
        connection.commit()

      return "Actualización exitosa"
  except psycopg2.Error as error:
      print("Error al actualizar compatibilidad:", error)
      connection.rollback()
      return None

      

################################################################################################
#################################### INSERTAR FEATURES CV ######################################
################################################################################################

def insertar_features_cv(features_cv, uniandes=False):

  features_cv['cluster'] = int(features_cv['cluster'])
  features_cv['experiencia'] = int(features_cv['experiencia'])

  connection = conectar_base_datos()
  cursor = connection.cursor()

  columns = ', '.join(features_cv.keys())
  placeholders = ', '.join(['%s'] * len(features_cv))
  
  if not uniandes:
    query = f"INSERT INTO public.features_cv ({columns}) VALUES ({placeholders})"
  else:
    query = f"INSERT INTO public.features_cv_uniandes ({columns}) VALUES ({placeholders})"

  values = list(features_cv.values())

  cursor.execute(query, values)
  connection.commit()
  cursor.close()
  connection.close()

################################################################################################
################################# INSERTAR FEATURES PROYECTOS ##################################
################################################################################################


def insertar_features_proyectos(features_proyectos, uniandes=False):

  features_proyectos['cluster'] = int(features_proyectos['cluster'])
  features_proyectos['experiencia'] = int(features_proyectos['experiencia'])
  connection = conectar_base_datos()
  cursor = connection.cursor()

  columns = ', '.join(features_proyectos.keys())
  placeholders = ', '.join(['%s'] * len(features_proyectos))

  if not uniandes:
    query = f"INSERT INTO public.features_projects ({columns}) VALUES ({placeholders})"
  else:
    query = f"INSERT INTO public.features_projects_uniandes ({columns}) VALUES ({placeholders})"

  values = list(features_proyectos.values())

  cursor.execute(query, values)
  connection.commit()
  cursor.close()
  connection.close()

################################################################################################
##################################### OBTENER VECTORES CV ######################################
################################################################################################

def obtener_vectores_cvs_cluster(cluster, uniandes=False):
  connection = conectar_base_datos()
  cursor = connection.cursor()

  if not uniandes:
    query = f"SELECT * FROM public.features_cv WHERE cluster = {cluster}".format(cluster)
  else:
    query = f"SELECT * FROM public.features_cv_uniandes WHERE cluster = {cluster}".format(cluster)

  cursor.execute(query)
  resultados = cursor.fetchall()
  cursor.close()
  connection.close()

  columnas = [desc[0] for desc in cursor.description]
  df_resultados = pd.DataFrame(resultados, columns=columnas)
  return df_resultados

def obtener_vectores_cvs_id(id, uniandes=False):
  connection = conectar_base_datos()
  cursor = connection.cursor()

  if not uniandes:
    query = f"SELECT * FROM public.features_cv WHERE id = '{id}'".format(id)
  else:
    query = f"SELECT * FROM public.features_cv_uniandes WHERE id = '{id}'".format(id)

  cursor.execute(query)
  resultados = cursor.fetchall()
  cursor.close()
  connection.close()

  columnas = [desc[0] for desc in cursor.description]
  df_resultados = pd.DataFrame(resultados, columns=columnas)
  return df_resultados.drop(['cluster', 'id', 'experiencia', 'softskills', 'hardskills', 'carrera'], axis = 1)
################################################################################################
################################# OBTENER VECTORES PROYECTOS ###################################
################################################################################################


def obtener_vectores_oportunidades_cluster(cluster, uniandes=False):
  connection = conectar_base_datos()
  cursor = connection.cursor()

  if not uniandes:
    query = f"SELECT * FROM public.features_projects WHERE cluster = {cluster}".format(cluster)
  else:
    query = f"SELECT * FROM public.features_projects_uniandes WHERE cluster = {cluster}".format(cluster)

  cursor.execute(query)
  resultados = cursor.fetchall()
  cursor.close()
  connection.close()

  columnas = [desc[0] for desc in cursor.description]
  df_resultados = pd.DataFrame(resultados, columns=columnas)
  return df_resultados

def obtener_vectores_oportunidades_id(id, uniandes=False):
  connection = conectar_base_datos()
  cursor = connection.cursor()

  if not uniandes:
    query = f"SELECT * FROM public.features_projects WHERE id = '{id}'".format(id)
  else:
    query = f"SELECT * FROM public.features_projects_uniandes WHERE id = '{id}'".format(id)

  cursor.execute(query)
  resultados = cursor.fetchall()
  cursor.close()
  connection.close()

  columnas = [desc[0] for desc in cursor.description]
  df_resultados = pd.DataFrame(resultados, columns=columnas)
  return df_resultados.drop(['cluster', 'id', 'experiencia', 'softskills', 'hardskills', 'carrera'], axis = 1)

################################################################################################
##################################### OBTENER VECTORES CV ######################################
################################################################################################


def obtener_features_cv(ids_cvs, uniandes):
  if len(ids_cvs) == 0:
    return []

  connection = conectar_base_datos()
  cursor = connection.cursor()

  ids_cvs = [f"'{id}'" for id in ids_cvs]
  ids_parametro = ', '.join(ids_cvs)

  if not uniandes:
    query = f"SELECT * FROM public.features_cv WHERE id in ({ids_parametro})".format(ids_parametro)
  else:
    query = f"SELECT * FROM public.features_cv_uniandes WHERE id in ({ids_parametro})".format(ids_parametro)

  cursor.execute(query)
  columnas = [desc[0] for desc in cursor.description]
  resultados = []
  for fila in cursor.fetchall():
      resultado = dict(zip(columnas, fila))
      resultados.append(resultado)

  cursor.close()
  connection.close()
  return resultados

################################################################################################
################################## OBTENER VECTORES PROYECTOS ##################################
################################################################################################

def obtener_features_proyectos(ids_proyectos, uniandes):

  if len(ids_proyectos) == 0:
    return []
  connection = conectar_base_datos()
  cursor = connection.cursor()
  ids_proyectos = [f"'{id}'" for id in ids_proyectos]
  ids_parametro = ', '.join(ids_proyectos)

  if not uniandes:
    query = f"SELECT * FROM public.features_projects WHERE id in ({ids_parametro})".format(ids_parametro)
  else:
    query = f"SELECT * FROM public.features_projects_uniandes WHERE id in ({ids_parametro})".format(ids_parametro)

  cursor.execute(query)
  columnas = [desc[0] for desc in cursor.description]
  resultados = []
  for fila in cursor.fetchall():
      resultado = dict(zip(columnas, fila))
      resultados.append(resultado)

  cursor.close()
  connection.close()
  return resultados

###### HASTA ACA ES LOCAL TOCA CUADRAR PGADMIN ####

def limpiar_espacio(cadena):
    s = cadena
    s = s.replace('\rn', ' ')
    s = s.replace('\t', ' ')
    s = s.replace('\f', ' ')
    s = s.replace('\n', ' ')
    s = s.replace('  ', ' ')
    return s

def procesar_texto(texto):
  # Traducir texto a español si está en inglés
  texto = limpiar_espacio(texto)
  patron_url = re.compile(r"http[s]?://\S+")
  texto_sin_urls = patron_url.sub('', texto)
  texto_sin_urls = texto_sin_urls.replace('_', ' ')
  try:
    resultado_traduccion = traductor.translate(texto_sin_urls, dest='es').text
  except Exception as e:
    print(e)
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

def procesar_texto_docs(corpus):
  if isinstance(corpus, str):
    return
  else:
    return [procesar_texto(texto) for texto in corpus]

def extraer_major(texto):
  texto_procesado = procesar_texto(texto)
  lista_majors = []

  for k,v in carreras_a_majors_procesado.items():
    if (len(k.split())> 1 and texto_procesado.find(k) != -1) or (len(k.split()) == 1 and k in texto_procesado.split()):
      lista_majors+=v.split()
  
  for key, values in keywords_procesado.items():
    if key in ' '.join(lista_majors):
      lista_majors+= values

  return list(lista_majors)

def extraer_soft_skills(texto):
  texto_procesado = procesar_texto(texto)
  lista_soft_skills = []

  for k,v in soft_skills_procesado.items():
    if (len(k.split())> 1 and texto_procesado.find(k) != -1) or (len(k.split()) == 1 and k in texto_procesado.split()):
      lista_soft_skills+=v.split()
      

  for key, values in keywords_procesado.items():
    if key in ' '.join(lista_soft_skills):
      lista_soft_skills+= values

  return list(set(lista_soft_skills))

def extraer_hard_skills(texto):
  texto_procesado = procesar_texto(texto)
  lista_hard_skills = []

  for k,v in hard_skills_procesado.items():
    if (len(k.split())> 1 and texto_procesado.find(k) != -1) or (len(k.split()) == 1 and k in texto_procesado.split()):
      lista_hard_skills+=v.split()

  for key, values in keywords_procesado.items():
    if key in ' '.join(lista_hard_skills):
      lista_hard_skills+= values

  return list(set(lista_hard_skills))

def descargar_data_cv(id_cv, uniandes =False):
  if uniandes:
    client = MongoClient("mongodb+srv://danielCTO:Coally2023-123@uniandescluster.h6u8ndo.mongodb.net/?retryWrites=true&w=majority&appName=UniandesCluster") 
    db = client['development']
  else:
    client = MongoClient("mongodb+srv://danielCTO:Coally2023-123@coally.nqokc.mongodb.net/CoallyProd?authSource=admin&replicaSet=atlas-39r1if-shard-0&w=majority&readPreference=primary&retryWrites=true&ssl=true")
    db = client['CoallyProd']

  db_cvs = db['usercvs']
  data_cv = db_cvs.find_one({'_id':ObjectId(str(id_cv))})

  lista_columnas = ["_id","educacion", "aptitudes_principales","experiencia", "extracto"]
  temp_dict = {}
  for elemento in lista_columnas:
    if elemento in data_cv.keys():
      temp_dict[elemento] = data_cv[elemento]
    elif elemento != 'experiencia':
      temp_dict[elemento] = ''
    else:
      temp_dict[elemento] = 0
  return temp_dict

def descargar_data_proyecto(id_proyecto, uniandes=False):

  if uniandes:
    client = MongoClient("mongodb+srv://danielCTO:Coally2023-123@uniandescluster.h6u8ndo.mongodb.net/?retryWrites=true&w=majority&appName=UniandesCluster")
    db = client['development']
  else:
    client = MongoClient("mongodb+srv://danielCTO:Coally2023-123@coally.nqokc.mongodb.net/CoallyProd?authSource=admin&replicaSet=atlas-39r1if-shard-0&w=majority&readPreference=primary&retryWrites=true&ssl=true")
    db = client['CoallyProd']
  
  db_proyectos = db['projects']
  data_proyecto = db_proyectos.find_one({'_id':ObjectId(id_proyecto)})

  lista_columnas = ["_id","NombreOportunidad", "DescribeProyecto", "municipio", "responsabilidadYfunciones", "country","habilidadesTecnicas","Niveldeconocimiento","experienciaAnos","habilidadesBlandas","empleos_alternativos","SeleccionaCarrera","departamento"]
  temp_dict = {}
  for elemento in lista_columnas:
    if elemento in data_proyecto.keys():
      temp_dict[elemento] = data_proyecto[elemento]
    elif elemento != 'experienciaAnos':
      temp_dict[elemento] = ''
    else:
      temp_dict[elemento] = 0
  return temp_dict

def transformar_data_cv(data_cv):
  temp_dict = {}
  temp_dict['_id'] = str(data_cv['_id'])
  temp_dict['aptitudes_principales'] = '~ '.join(data_cv['aptitudes_principales']) if data_cv['aptitudes_principales'] != None else ''
  temp_dict['Titulos'] = (lambda x: ', '.join([i['Titulo_Certificacion'] for i in x if x is not None]))(data_cv['educacion']) if data_cv['educacion'] is not None else ''
  temp_dict['Instituciones'] = (lambda x: ', '.join([i['NombreInstitucion'] for i in x]))(data_cv['educacion']) if data_cv['educacion'] is not None else ''
  temp_dict['cargos'] = (lambda x: '~ '.join([cargo['nombrecargo'] for experiencia in x for cargo in experiencia.get('cargos', []) if x is not None and experiencia.get('cargos') is not None and cargo['nombrecargo'] != '-']))(data_cv.get('experiencia', [])) if data_cv.get('experiencia') is not None else ''
  temp_dict['experiencia'] = (lambda x: sum([experiencia['totalDuracion'] for experiencia in x if 'totalDuracion' in experiencia.keys() and type(experiencia['totalDuracion']) != str]))(data_cv['experiencia']) if data_cv['experiencia'] != None else 0
  for k,v in data_cv.items():
    if k not in temp_dict.keys():
      temp_dict[k] = v
  return temp_dict

def transformar_data_proyecto(data_proyecto):

  def procesar_experiencia_proyecto(x):
    try:
      return float(x)*12
    except:
      if x.find('-') != -1:
        return sum(map(float,x.split('-')))/len(x.split('-'))*12
      if x == '':
        return 0
    return 0

  temp_dict = {}
  temp_dict['_id'] = str(data_proyecto['_id'])
  temp_dict['responsabilidadYfunciones'] = '~ '.join(data_proyecto['responsabilidadYfunciones']) if data_proyecto['responsabilidadYfunciones'] != None else ''
  temp_dict['Niveldeconocimiento'] = '~ '.join(data_proyecto['Niveldeconocimiento']) if data_proyecto['Niveldeconocimiento'] != None else ''
  temp_dict['empleos_alternativos'] = '~ '.join(data_proyecto['empleos_alternativos']) if data_proyecto['empleos_alternativos'] != None else ''
  temp_dict['habilidadesTecnicas'] = '~ '.join(data_proyecto['habilidadesTecnicas']) if data_proyecto['habilidadesTecnicas'] != None else ''
  temp_dict['habilidadesBlandas'] = '~ '.join([i['name'] if isinstance(i, dict) else i for i in data_proyecto['habilidadesBlandas'] ])
  for k,v in data_proyecto.items():
    if k not in temp_dict.keys():
      temp_dict[k] = v
  temp_dict['experiencia_meses'] = procesar_experiencia_proyecto(temp_dict['experienciaAnos'])
  return temp_dict

def extraer_features_cv(data_cv_transformada):
  dict_features = {}
  dict_features['id'] = data_cv_transformada['_id']

  dict_features['experiencia'] = data_cv_transformada['experiencia']
  
  data_cv_transformada['descripcion_carrera'] = data_cv_transformada['extracto']+' '+ data_cv_transformada['Titulos'] + data_cv_transformada['cargos']
  dict_features['carrera'] = (lambda x: ' '.join(extraer_major(x)))(data_cv_transformada['descripcion_carrera'])

  data_cv_transformada['descripcion_softskills'] = data_cv_transformada['extracto']+' '+ data_cv_transformada['aptitudes_principales'] + data_cv_transformada['cargos']
  dict_features['softskills'] = (lambda x: ' '.join(extraer_soft_skills(x)))(data_cv_transformada['descripcion_softskills'])

  data_cv_transformada['descripcion_hardskills'] = data_cv_transformada['extracto']+' '+ data_cv_transformada['aptitudes_principales'] + data_cv_transformada['cargos']
  dict_features['hardskills'] = (lambda x: ' '.join(extraer_hard_skills(x)))(data_cv_transformada['descripcion_hardskills'])
  return dict_features

def extraer_features_proyecto(data_proyecto_transformada):
  dict_features = {}
  dict_features['id'] = data_proyecto_transformada['_id']

  dict_features['experiencia'] = data_proyecto_transformada['experiencia_meses']

  data_proyecto_transformada['descripcion_carrera'] = data_proyecto_transformada['DescribeProyecto']+' '+data_proyecto_transformada['NombreOportunidad']+' '+data_proyecto_transformada['habilidadesTecnicas']+' '+data_proyecto_transformada['SeleccionaCarrera']+' '+data_proyecto_transformada['empleos_alternativos']
  dict_features['carrera'] = (lambda x: ' '.join(extraer_major(x)))(data_proyecto_transformada['descripcion_carrera'])
  if 'sistem' in dict_features['carrera'] and 'ingenieri' in dict_features['carrera']:
    dict_features['carrera'] += 'frontend desarroll backend'

  data_proyecto_transformada['descripcion_softskills'] = data_proyecto_transformada['DescribeProyecto']+' '+data_proyecto_transformada['habilidadesBlandas']
  dict_features['softskills'] = (lambda x: ' '.join(extraer_soft_skills(x)))(data_proyecto_transformada['descripcion_softskills'])

  data_proyecto_transformada['descripcion_hardskills'] = data_proyecto_transformada['DescribeProyecto']+' '+data_proyecto_transformada['habilidadesTecnicas']
  dict_features['hardskills'] = (lambda x: ' '.join(extraer_hard_skills(x)))(data_proyecto_transformada['descripcion_hardskills'])
  if 'program' in dict_features['hardskills']:
    dict_features['hardskills'] += 'frontend desarroll backend'
  return dict_features

def clusterizar(descripcion):
  if type(descripcion) == str:
    descripcion = [descripcion]
  vector = pipeline.named_steps['preprocess'].transform(descripcion)
  cluster = pipeline.predict(descripcion)[0]
  return cluster, vector[0]

def obtener_mejores_oportunidades_similitud(cluster, vector_cv, uniandes=False):
  vectores_oportunidades_cluster = obtener_vectores_oportunidades_cluster(cluster, uniandes)
  if len(vectores_oportunidades_cluster) == 0:
    return {}
  similitud_cos = cosine_similarity(vector_cv, vectores_oportunidades_cluster.drop(['cluster', 'id', 'experiencia', 'softskills', 'hardskills', 'carrera'], axis = 1))
  indices = np.where(similitud_cos[0] > 0.2)[0]
  return dict(zip(vectores_oportunidades_cluster.iloc[indices]['id'], similitud_cos[0][indices]))

def obtener_mejores_cvs_similitud(cluster, vector_oportunidad, uniandes=False):
  vectores_cvs_cluster = obtener_vectores_cvs_cluster(cluster, uniandes)
  if len(vectores_cvs_cluster) == 0:
    return {}
  similitud_cos = cosine_similarity(vector_oportunidad, vectores_cvs_cluster.drop(['cluster', 'id', 'experiencia', 'softskills', 'hardskills', 'carrera'], axis = 1))
  indices = np.where(similitud_cos[0] > 0.2)[0]
  return dict(zip(vectores_cvs_cluster.iloc[indices]['id'], similitud_cos[0][indices]))

def calcular_features(features_cv, features_proyecto):
  # tech skills
  tech_skills_cv = features_cv['hardskills'].split()
  tech_skills_proyecto = features_proyecto['hardskills'].split()
  if len(tech_skills_proyecto) > 0:
    porcentaje_tech = len([i for i in tech_skills_cv if i in tech_skills_proyecto])/len(tech_skills_proyecto)
  else:
    porcentaje_tech = 0

  # soft skills
  soft_skills_cv = features_cv['softskills'].split()
  soft_skills_proyecto = features_proyecto['softskills'].split()
  if len(soft_skills_proyecto) > 0:
    porcentaje_soft = len([i for i in soft_skills_cv if i in soft_skills_proyecto])/len(soft_skills_proyecto)
  else:
    porcentaje_soft = 0

  # carrera
  carrera_cv = features_cv['carrera'].split()
  carrera_proyecto = features_proyecto['carrera'].split()
  if len(carrera_proyecto) > 0:
    porcentaje_carrera = len([i for i in carrera_cv if i in carrera_proyecto])/len(carrera_proyecto)
  else:
    porcentaje_carrera = 0

  if not porcentaje_tech:
    porcentaje_tech = 0

  if not porcentaje_soft:
    porcentaje_soft = 0

  if not porcentaje_carrera:
    porcentaje_carrera = 0

  return porcentaje_tech, porcentaje_carrera

def calcular_porcentaje_similitud(features):
  modelo.predict_proba(features)

def agregar_cv(id_cv, uniandes=False):
  data_cv = descargar_data_cv(id_cv, uniandes)
  data_cv_transformada = transformar_data_cv(data_cv)
  features_cv = extraer_features_cv(data_cv_transformada)
  cluster, vector = clusterizar((data_cv_transformada['aptitudes_principales']+' '+data_cv_transformada['extracto']+' '+data_cv_transformada['Titulos']).replace('~',','))
  features_cv['cluster'] = cluster

  for index, item in enumerate(vector):
    features_cv['x'+str(index+1)] = item

  insertar_features_cv(features_cv, uniandes)
  mejores_oportunidades_similitud = obtener_mejores_oportunidades_similitud(cluster, [vector], uniandes)
  ids = mejores_oportunidades_similitud.keys()
  lista_features = obtener_features_proyectos(ids, uniandes)
  connection = conectar_base_datos()
  cursor = connection.cursor()
  if connection is None:
      return None

  
  for features, similitud in zip(lista_features, mejores_oportunidades_similitud.values()):
    id = features['id']

    if( features_cv['experiencia'] == 0) or features['experiencia'] ==0:
      relacion_experiencia = 0
    else:
      relacion_experiencia = features_cv['experiencia']/features['experiencia']

    features_finales = [relacion_experiencia]+list(calcular_features(features_cv, features)) + [similitud]

    X = pd.DataFrame({k:[v] for k,v in zip(['relacion_experiencia', 'porcentaje_tech', 'porcentaje_carrera', 'similitud'],features_finales)})

    X_scaled = scaler.transform(X)
    compatibilidad = modelo.predict_proba(X_scaled)[0]*min(similitud/0.6, 1)

    actualizar_compatibilidad(connection, cursor, compatibilidad, id_cv, id, uniandes)
   
  connection.close()
  cursor.close()
  return 200

def agregar_proyecto(id_proyecto, uniandes=False):
  data_proyecto = descargar_data_proyecto(id_proyecto, uniandes)
  data_proyecto_transformada = transformar_data_proyecto(data_proyecto)
  features_proyecto = extraer_features_proyecto(data_proyecto_transformada)
  cluster, vector = clusterizar((data_proyecto_transformada['NombreOportunidad']+' '+data_proyecto_transformada['DescribeProyecto']+' '+data_proyecto_transformada['responsabilidadYfunciones']+' '+data_proyecto_transformada['habilidadesTecnicas']+' '+data_proyecto_transformada['habilidadesBlandas']+' '+data_proyecto_transformada['empleos_alternativos']+' '+data_proyecto_transformada['SeleccionaCarrera']).replace('~',','))
  features_proyecto['cluster'] = cluster
  for index, item in enumerate(vector):
    features_proyecto['x'+str(index+1)] = item
  insertar_features_proyectos(features_proyecto, uniandes)
  mejores_cvs_similitud = obtener_mejores_cvs_similitud(cluster, [vector], uniandes)
  ids = mejores_cvs_similitud.keys()
  lista_features = obtener_features_cv(ids, uniandes)
  connection = conectar_base_datos()
  cursor = connection.cursor()

  if connection is None:
      return None

  for features, similitud in zip(lista_features, mejores_cvs_similitud.values()):
    id = features['id']

    if( features_proyecto['experiencia'] == 0) or features['experiencia'] ==0:
      relacion_experiencia = 0
    else:
      relacion_experiencia = features['experiencia']/features_proyecto['experiencia']


    features_finales = [relacion_experiencia]+list(calcular_features(features, features_proyecto)) + [similitud]

    X = pd.DataFrame({k:[v] for k,v in zip(['relacion_experiencia', 'porcentaje_tech', 'porcentaje_carrera', 'similitud'],features_finales)})
    X_scaled = scaler.transform(X)
    compatibilidad = modelo.predict_proba(X_scaled)[0]*min(similitud/0.6, 1)

    actualizar_compatibilidad(connection, cursor, compatibilidad, id, id_proyecto, uniandes)
  return 200

def agregar_aplicante(id_job, id_cv, uniandes = False):
  features_cv = obtener_features_cv([id_cv], uniandes)[0]
  features_proyecto = obtener_features_proyectos([id_job], uniandes)[0]

  features = calcular_features(features_cv, features_proyecto)

  if( features_proyecto['experiencia'] == 0) or features_cv['experiencia'] ==0:
    relacion_experiencia = 0
  else:
    relacion_experiencia = features_cv['experiencia']/features_proyecto['experiencia']


  vector_cv = obtener_vectores_cvs_id(id_cv, uniandes)
  vector_oportunidad =obtener_vectores_oportunidades_id(id_job, uniandes)

  similitud_cos = cosine_similarity(vector_cv, vector_oportunidad)

  features_finales = [relacion_experiencia]+list(features) + list(similitud_cos[0])

  X = pd.DataFrame({k:[v] for k,v in zip(['relacion_experiencia', 'porcentaje_tech', 'porcentaje_carrera', 'similitud'],features_finales)})
  X_scaled = scaler.transform(X)

  cv_df = pd.DataFrame({k:[v] for k,v in features_cv.items()})
  oportunidad_df = pd.DataFrame({k:[v] for k,v in features_proyecto.items()})

  similitud = cosine_similarity(cv_df.drop(['cluster', 'id', 'experiencia', 'softskills', 'hardskills', 'carrera'], axis = 1), oportunidad_df.drop(['cluster', 'id', 'experiencia', 'softskills', 'hardskills', 'carrera'], axis = 1))

  compatibilidad = max(modelo.predict_proba(X_scaled)[0]*min(similitud[0][0]/0.6, 1),0)

  connection = conectar_base_datos()
  cursor = connection.cursor()

  if connection is None:
      return None
  
  try:
      if not uniandes:
        cursor.execute(f"""DELETE FROM public.general_compatibility WHERE id_resume = '{id_cv}' AND id_job = '{id_job}' """.format(id_job=id_job,  id_cv=id_cv))
        connection.commit()
      else:
        cursor.execute(f"""DELETE FROM public.general_compatibility_uniandes WHERE id_resume = '{id_cv}' AND id_job = '{id_job}' """.format(id_job=id_job, id_cv=id_cv))
        connection.commit()

  except psycopg2.Error as error:
      print("Error al actualizar compatibilidad:", error)
      connection.rollback()
      return None
  
  actualizar_compatibilidad(connection, cursor, compatibilidad, id_cv, id_job, uniandes)
  return 200


#########################################################
##################### API REST ##########################
#########################################################

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/test")
async def test():
 return "Hello World!"

@app.post("/api/create_cv")
def create_cv(request: CreateCVRequestModel):
  try:
    id_cv = request.id_cv
    uniandes = request.uniandes

    try:
      agregar_cv(id_cv, uniandes)
    except:
      agregar_cv(id_cv, not uniandes)
  except:
    return

@app.post("/api/create_project")
def create_project(request: CreateProjectRequestModel):
  try:
    id_project = request.id_project
    uniandes = request.uniandes

    try:
      agregar_proyecto(id_project, uniandes)
    except:
      agregar_proyecto(id_project, not uniandes)
  except:
    return

@app.post("/api/add_applicant")
def add_applicant(request: AddApplicantRequestModel):
  try:
    id_project = request.id_project
    id_cv = request.id_cv
    uniandes = request.uniandes

    agregar_aplicante(id_project, id_cv, uniandes)
  except:
    return