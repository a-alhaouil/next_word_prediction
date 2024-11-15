import time

from joblib import load, dump
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import threading
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec





app = Flask(__name__)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Load LSTM model and vocabulary
lstm_model = tf.keras.models.load_model("NWP-USE.keras")
vocabulary = np.load('vocabulary.npy', allow_pickle=True)
vocab_dict = {word: i for i, word in enumerate(vocabulary)}

module_url = r"C:\Users\abdes\Desktop\predict_search\universal_model_encoder_tf\063d866c06683311b44b4992fd46003be952409c"
embed = hub.load(module_url)
# embed = (hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4"))

# Load TF-IDF vectorizer and Multinomial Naive Bayes model
tfidf_vectorizer = load('tfidf_vectorizer.joblib')
nb_model = load('nb_model.joblib')


# # Load AdaBoost model and vocabulary for Word2Vec
# ada_boost_model = load('./ada_boost_model.pkl')
#
# vocabulary_word2vec = np.load('vocabulary_fo_rwr2vc.npy', allow_pickle=True)
#
# # Load Word2Vec model and AdaBoost model
# word2vec_model = Word2Vec.load('./word2vec_model')

def predict_next_word(sentence, model):
    if model == 'LSTM':
        return predict_next_word_lstm(sentence)
    elif model == 'TF-IDF':
        return predict_next_word_tfidf(sentence)
    elif model == 'Cosine-Adaboost':
        return predict_next_word_cosine_adaboost(sentence)
    elif model == 'LSA':
        return predict_next_word_lsa(sentence)
    else:
        raise ValueError(f"Model '{model}' not found")


def predict_next_word_lstm(sentence):
    embedding = embed([sentence]).numpy()
    prediction = lstm_model.predict(embedding)
    next_word_idx = np.argmax(prediction[-1])
    next_word = vocabulary[next_word_idx]
    return next_word

def predict_next_word_tfidf(sentence):
    vectorized_sentence = tfidf_vectorizer.transform([sentence])
    prediction = nb_model.predict_proba(vectorized_sentence)[0]
    next_word_idx = np.argmax(prediction)
    next_word = vocabulary[next_word_idx]
    return next_word


def preprocess_sentence(sentence):
    # Clean the sentence by lowering the case and filtering out non-alphabetic characters
    return [word.lower() for word in sentence.split() if word.isalpha()]


def predict_next_word_cosine_adaboost(sentence):
    # # Preprocess the sentence (lowercase and remove punctuation)
    # words = preprocess_sentence(sentence)
    #
    # # Check which words in the sentence are in the vocabulary
    # word_embeddings = [word for word in words if word in vocabulary_word2vec]
    #
    # # Debugging: print out the found word embeddings
    # print("Word Embeddings Found:", word_embeddings)
    #
    # if not word_embeddings:
    #     return "No relevant words found in vocabulary"
    #
    # # Get the embeddings for the words in the sentence
    # sentence_embedding = np.mean([word2vec_model.wv[word] for word in word_embeddings], axis=0).reshape(1, -1)
    #
    # # Debugging: print out the sentence embedding shape
    # print("Sentence Embedding Shape:", sentence_embedding.shape)
    #
    # # Calculate cosine similarity between sentence embedding and all words in the vocabulary
    # scores = [
    #     cosine_similarity(sentence_embedding, word2vec_model.wv[word].reshape(1, -1))[0][0]
    #     for word in vocabulary_word2vec
    # ]
    #
    # # Debugging: print out the scores for each word
    # print("Cosine Similarity Scores:", scores)
    #
    # scores_array = np.array(scores).reshape(1, -1)
    #
    # # Predict the next word based on the scores
    # predicted_idx = ada_boost_model.predict(scores_array)[0]
    # print("Predicted Index:", predicted_idx)
    #
    # if 0 <= predicted_idx < len(vocabulary_word2vec):
    #     return vocabulary_word2vec[predicted_idx]
    # else:
    #     return "Prediction index out of range"
    return "predicted_word_adaboost"


def predict_next_word_lsa(sentence):
    # Implement LSA model logic for predicting next word
    return "predicted_word_lsa"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    model = data['model']
    next_word = predict_next_word(text, model)
    return jsonify({'next_word': next_word})




# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)







# def run_app():
#     server_thread = threading.Thread(target=app.run)
#     server_thread.start()
#
#     # Wait for 60 seconds
#     time.sleep(60)
#
#     # Stop the Flask server
#     func = request.environ.get('werkzeug.server.shutdown')
#     if func is not None:
#         func()
#
#     server_thread.join()
#
# threading.Thread(target=run_app).start()