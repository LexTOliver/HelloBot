import random
import json
import pickle5 as pickle
import numpy as np

import spacy

from tensorflow.keras.models import load_model

nlp = spacy.load("pt_core_news_sm")
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
  tokens = nlp(sentence)
  sentence_words = [w.lemma_.lower() for w in tokens]
  return sentence_words

def bag_of_words(sentence):
  sentence_words = clean_up_sentence(sentence)
  bag = [0] * len(words)
  for w in sentence_words:
    for i, word in enumerate(words):
      if word == w:
        bag[i] = 1
  return np.array(bag)

def predict_class(sentence):
  bow = bag_of_words(sentence)
  res = model.predict(np.array([bow]))[0]
  ERROR_THRESHOLD = 0.25
  results = [[i,r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]

  results.sort(key=lambda x:x[1], reverse=True)
  return_list = []
  for r in results:
    return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

  return return_list

def get_response(intents_list, intents_json):
  tag = intents_list[0]['intent']
  
  for i in intents_json['intents']:
    if i['tag'] == tag:
      response = random.choice(i['responses'])
      break

  return response

print('GO! Bot is running!')

while True:
  message = input('')
  ints = predict_class(message)
  res = get_response(ints, intents)
  print(res)