#El chatbot predice la categoría a la que pertenece la entrada del usuario 
#y luego genera una respuesta aleatoria de una lista de respuestas predeterminadas
#para esa categoría. 

#Librerías
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

#Sintesis de audio
import speech_recognition as sr
import pyttsx3

#Modelo generador de Parráfos
#from s05_RNN_ParagrapsGeneratorPredict4 import predict_paragraph
from s06_URL_Search_02 import busqueda
#from s06_wikipedia import lee_wiki
#from s06_GoogleSearchAPI import buscar_informacion
#from s06_StopWords import eliminar_palabras_genericas

nltk.download('punkt_tab')

#Aplica la lematización (forma raiz)
lemmatizer = WordNetLemmatizer()

#Importar modelo y los archivos base
print('\nImportar modelo (.keras) y los archivos serializados (.pkl) ...')

with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('agilbot_model.keras')

#Función. Tokeniza una oración (la divide en palabras) y luego aplica la lematización (forma raiz)
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#Codificación de la Oración (Bag of Words)
#Convierte la información a una representación binaria
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    #print(bag)
    return np.array(bag)

#Predice la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res ==np.max(res))[0][0]
    category = classes[max_index]
    return category

#Genera una respuesta aleatoria
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"]==tag:
            result = random.choice(i['responses'])
            break
    return result

def respuesta(message):
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res, ints


#Sintesis de audio
# opciones de voz /idioma
id1 = 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0'

# Inicializar el motor de síntesis de voz
engine = pyttsx3.init()
engine.setProperty('voice', id1)

# Configurar propiedades de la voz (opcional)
engine.setProperty('rate', 150)     # Velocidad de la voz
engine.setProperty('volume', 0.9)   # Volumen de la voz


# Ejecución ChatBot
while True:
    message = input('Ingresa tu consulta: ').lower()  # Convertir a minúsculas para mayor flexibilidad
    
    # Condición para salir si el usuario escribe "adiós" o alguna variante
    if message in ['adiós', 'adios', 'bye', 'hasta luego', 'nos vemos']:
        print('[ AGILBOT ]: ¡Hasta Pronto! Fue un placer ayudarte, recuerda que estoy aqui para resolver tus dudas sobre proyectos ágiles.')
        engine.say("¡Hasta Pronto! Fue un placer ayudarte, recuerda que estoy aqui para resolver tus dudas sobre proyectos ágiles.")
        engine.runAndWait()
        break  # Salir del bucle
    
    # Generar respuesta del chatbot
    my_message = message
    text, tag = respuesta(message)
    
    print('[ AGILBOT ]: ' + text + '\n')

    # Hacer que el motor hable
    engine.say(text)
    engine.runAndWait()

    #Tags validos para busqueda online 
    tags_validos = ['tecnología','programación','proyecto ágil','desarrollo ágil','sprint','backlog','tarea en un proyecto ágil','herramientas de gestión de proyectos']
    if tag in tags_validos:
        #Paginas recomendadas
        engine.say("Puedes buscar información en las siguientes páginas web: ")
        engine.runAndWait()
        busqueda(text,3)



