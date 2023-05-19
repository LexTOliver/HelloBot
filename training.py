import random
import json
import pickle5 as pickle
import numpy as np

import spacy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD


# LOADING TRAINING DATA (from intents.json)

nlp = spacy.load("pt_core_news_sm")

intents = json.loads(open("intents.json").read())

words = []  #words = list of all lemmatized words from intents patterns
classes = []  #classes = list of intents tags
documents = []  #documents = list of tuple with words and respective tags
ignore_puncts = ['?', '!', '.', ',']

for intent in intents['intents']:
  if intent['tag'] not in classes:
    classes.append(intent['tag'])
  for pattern in intent['patterns']:
    tokens = nlp(pattern)
    words.extend(tokens)
    documents.append((tokens, intent['tag']))

words = [w.lemma_.lower() for w in words if w.text not in ignore_puncts]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# PREPARING TRAINING DATA (bag of words)

training = []
output_empty = [0] * len(classes)

for document in documents:
  bag = []
  word_patterns = document[0]
  word_patterns = [w.lemma_.lower() for w in word_patterns]
  for word in words:
    bag.append(1) if word in word_patterns else bag.append(0)
  
  output_row = list(output_empty)
  output_row[classes.index(document[1])] = 1
  training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])


# CREATING CHATBOT MODEL (TODO: Estudar topologias/arquiteturas de redes neurais e suas camadas)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Done!')