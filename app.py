from sklearn.feature_extraction.text import CountVectorizer
import csv
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import requests
# pprint is used to format the JSON response
from pprint import pprint
import os
import pandas as pd
import numpy as np
from flask import Flask, request, abort, jsonify
import math

subscription_key = "7c59cfce389144619efad75105598f30"
endpoint = "https://cpftext.cognitiveservices.azure.com/"

sentiment_url = endpoint + "/text/analytics/v2.1/sentiment"
language_api_url = endpoint + "/text/analytics/v2.1/languages"
keyphrase_url = endpoint + "/text/analytics/v2.1/keyphrases"

freq_u1 = {}
with open('u1_word_frequencies.csv', mode='r') as infile:
    reader = csv.reader(infile)
    freq_u1 = {rows[0]:int(rows[1]) for rows in reader}

freq_u2 = {}
with open('u2_word_frequencies.csv', mode='r') as infile:
    reader = csv.reader(infile)
    freq_u2 = {rows[0]:int(rows[1]) for rows in reader}

freq_u3 = {}
with open('u3_word_frequencies.csv', mode='r') as infile:
    reader = csv.reader(infile)
    freq_u3 = {rows[0]:int(rows[1]) for rows in reader}

freq_u4 = {}
with open('u4_word_frequencies.csv', mode='r') as infile:
    reader = csv.reader(infile)
    freq_u4 = {rows[0]:int(rows[1]) for rows in reader}

freq_u5 = {}
with open('u5_word_frequencies.csv', mode='r') as infile:
    reader = csv.reader(infile)
    freq_u5 = {rows[0]:int(rows[1]) for rows in reader}

total_cnts_features = {}
with open('total_cnts_features.csv', mode='r') as infile:
    reader = csv.reader(infile)
    total_cnts_features = {rows[0]:rows[1] for rows in reader}

total_features = int(total_cnts_features['total'])
total_cnts_features_u1 = int(total_cnts_features['u1'])
total_cnts_features_u2 = int(total_cnts_features['u2'])
total_cnts_features_u3 = int(total_cnts_features['u3'])
total_cnts_features_u4 = int(total_cnts_features['u4'])
total_cnts_features_u5 = int(total_cnts_features['u5'])
nltk.download('punkt')

def predict_urgency(sentence):
    new_word_list = word_tokenize(sentence)
    print("hi")
    u_probabilities = {}
    with open('u_probabilities.csv', mode='r') as infile:
        reader = csv.reader(infile)
        u_probabilities = {rows[0]:rows[1] for rows in reader}
    print(u_probabilities)
    u1_prob = int(u_probabilities['u1'])
    u2_prob = int(u_probabilities['u2'])
    u3_prob = int(u_probabilities['u3'])
    u4_prob = int(u_probabilities['u4'])
    u5_prob = int(u_probabilities['u5'])

    prob_u1_with_ls = []
    for word in new_word_list:
        if word in freq_u1.keys():
            count = freq_u1[word]
        else:
            count = 0
        prob_u1_with_ls.append((count + 1)/(total_cnts_features_u1 + total_features))
    u1_dict = dict(zip(new_word_list,prob_u1_with_ls))
    for keyword in new_word_list:
        u1_prob = u1_prob * u1_dict[keyword]

    prob_u2_with_ls = []
    for word in new_word_list:
        if word in freq_u2.keys():
            count = freq_u2[word]
        else:
            count = 0
        prob_u2_with_ls.append((count + 1)/(total_cnts_features_u2 + total_features))
    u2_dict = dict(zip(new_word_list,prob_u2_with_ls))
    for keyword in new_word_list:
        u2_prob = u2_prob * u2_dict[keyword]

    prob_u3_with_ls = []
    for word in new_word_list:
        if word in freq_u3.keys():
            count = freq_u3[word]
        else:
            count = 0
        prob_u3_with_ls.append((count + 1)/(total_cnts_features_u3 + total_features))
    u3_dict = dict(zip(new_word_list,prob_u3_with_ls))
    for keyword in new_word_list:
        u3_prob = u3_prob * u3_dict[keyword]

    prob_u4_with_ls = []
    for word in new_word_list:
        if word in freq_u4.keys():
            count = freq_u4[word]
        else:
            count = 0
        prob_u4_with_ls.append((count + 1)/(total_cnts_features_u4 + total_features))

    u4_dict = dict(zip(new_word_list,prob_u4_with_ls))
    for keyword in new_word_list:
        u4_prob = u4_prob * u4_dict[keyword]
        
    prob_u5_with_ls = []
    for word in new_word_list:
        if word in freq_u5.keys():
            count = freq_u5[word]
        else:
            count = 0
        prob_u5_with_ls.append((count + 1)/(total_cnts_features_u5 + total_features))
        
    u5_dict = dict(zip(new_word_list,prob_u5_with_ls))
    for keyword in new_word_list:
        u5_prob = u5_prob * u5_dict[keyword]

    max_prob = max(u1_prob, u2_prob, u3_prob, u4_prob, u5_prob)
    if max_prob == u1_prob:
        return 0
    elif max_prob == u2_prob:
        return 1
    elif max_prob == u3_prob:
        return 2
    elif max_prob == u4_prob:
        return 3
    return 4
    
def extract_keywords(sentence):
    documents = {"documents": [
    {"id": "1", "language": "en",
        "text": sentence}]}
    
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    response = requests.post(keyphrase_url, headers=headers, json=documents)
    key_phrases = response.json()
    
    s = ""
    for j in range(len(response.json()['documents'][0]['keyPhrases'])):
        s = s + " " + response.json()['documents'][0]['keyPhrases'][j]
    if s == "":
        s = "NoKeywordsFound"
    return s

def extract_sentiment(sentence):
    documents = {"documents": [
    {"id": "1", "language": "en",
        "text": sentence}]}
    
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    response = requests.post(sentiment_url, headers=headers, json=documents)
    key_phrases = response.json()
    
    return response.json()['documents'][0]['score']

def get_urgency(x1):  
    print("Statement: " + x1)
    keys = extract_keywords(x1)
    print("Keywords: " + keys)
    sentiment = extract_sentiment(x1)
    print("Sentiment: " + str(sentiment))
    urgency_level = predict_urgency(keys)
    print("Original urgency: " + str(urgency_level))
    
    # Sentiment is between 0 and 1, higher score means more positive sentiment. So, higher score means less urgent
    urgency_level_final = round(urgency_level + ((1 - sentiment) * 5))
    print("Urgency with sentiment: " + str(urgency_level_final))
    print("-------------------------")
    return x1, keys, str(sentiment), str(urgency_level_final)


app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcome to CPF Urgency Prediction Service!"

@app.route("/urgency", methods=["POST"])
def urgency():
    urgency = request.json.get('sentence', None)
    if urgency is None:
        abort(403)
    else:
        result_tuple = get_urgency(urgency)
        return jsonify({
            'status': 'OK',
            'statement': result_tuple[0],
            'keywords': result_tuple[1],
            'sentiment': result_tuple[2],
            'urgency': result_tuple[3],
        })
        
if __name__ == '__main__':
    app.run(debug=False, port=8668)
