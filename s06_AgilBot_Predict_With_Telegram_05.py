# Importamos las librerías necesarias
import telebot
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Sintesis de audio (opcional si lo quieres en Telegram)
import pyttsx3

# Librerías para funciones adicionales
#from s05_RNN_ParagrapsGeneratorPredict4 import predict_paragraph
from s06_URL_Search_02 import busqueda

# Descargar el paquete necesario de nltk
nltk.download('punkt')

# Aplica la lematización (forma raíz)
lemmatizer = WordNetLemmatizer()

# Cargar el modelo y los archivos serializados
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('agilbot_model.keras')

# Función para limpiar y tokenizar la oración
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Codificación Bag of Words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predicción de la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

# Genera una respuesta aleatoria basada en la categoría predicha
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i['responses'])
            break
    return result

def respuesta(message):
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res, ints

# Configuración del bot de Telegram
API_TOKEN = '7595010058:AAFpj7fz754ZRdZCQEbkFlpZ3cEPdui4jDo'
bot = telebot.TeleBot(API_TOKEN)

# Función que recibe los mensajes de Telegram
@bot.message_handler(func=lambda message: True)
def chatbot_response(message):
    user_message = message.text.lower()  # Convertir a minúsculas
    if user_message in ['adiós', 'adios', 'bye', 'hasta luego', 'nos vemos']:
        farewell_message = "¡Hasta Pronto! Fue un placer ayudarte, recuerda que estoy aquí para resolver tus dudas sobre proyectos ágiles."
        bot.reply_to(message, farewell_message)
    else:
        # Predicción de la categoría y generación de respuesta
        text, tag = respuesta(user_message)
        bot.reply_to(message, text)

        # Opcional: realizar búsqueda online si la categoría coincide con los tags válidos
        tags_validos = ['tecnología', 'programación', 'proyecto ágil', 'desarrollo ágil', 'sprint', 'backlog', 'tarea en un proyecto ágil', 'herramientas de gestión de proyectos']
        if tag in tags_validos:
            bot.reply_to(message, "Puedes buscar información en las siguientes páginas web:")
            busqueda(text, 3)

# Iniciar el bot
print("El chatbot está listo y conectado a Telegram.")
bot.polling()
