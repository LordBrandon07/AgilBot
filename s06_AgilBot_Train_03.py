#Entrena un chatbot utilizando una red neuronal para clasificar patrones de texto 
# en categorías específicas basadas en un archivo de intents.json :: intenciones del usuario

#Librerías
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer #Para pasar las palabras a su forma raíz
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD 

#Cambia '/path/to/nltk_data'
nltk.data.path.append('D:\\DonBrandon\\Programing\\IA\\Bot\\nltk_data')  

#Reducir las palabras a su forma raíz (lemmatizer)
lemmatizer = WordNetLemmatizer()

#patrones de frases y respuestas categorizadas 
#en intents: intenciones del usuario
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

#Inicialización de listas
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

#Clasifica los patrones y las categorías
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)  # Aquí se utiliza word_tokenize
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

#Las palabras se convierten a su forma raíz, se eliminan duplicados, se guardan en archivos .pkl para su uso futuro.
#Las clases también se serializan.
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#Pasa la información a unos y ceros según las palabras presentes en cada categoría 
#para hacer el entrenamiento. Para cada documento se genera una "bolsa de palabras"
training = []
output_empty = [0]*len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
random.shuffle(training)
print(len(training)) 
train_x=[]
train_y=[]
for i in training:
    train_x.append(i[0])
    train_y.append(i[1])

train_x = np.array(train_x) 
train_y = np.array(train_y)

#Creación de la Red Neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), name="inp_layer", activation='relu'))
model.add(Dropout(0.5, name="hidden_layer1"))
model.add(Dense(64, name="hidden_layer2", activation='relu'))
model.add(Dropout(0.5, name="hidden_layer3"))
model.add(Dense(len(train_y[0]), name="output_layer", activation='softmax'))

#Compilación y Entrenamiento del Modelo
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True) #gradiente estocástico descendente
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#Entrenamos el modelo y lo guardamos
model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
model.save("agilbot_model.keras")
