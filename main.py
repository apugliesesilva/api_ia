import nltk
from flask import Flask, request, jsonify
import requests
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import os

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/ggrazzioli/cls_sentimento_sebrae"
headers = {"Authorization": "Bearer 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"}

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(comment):
    tokens = word_tokenize(comment.lower())
    tokens = [token for token in tokens if token not in string.punctuation and token not in stopwords.words(
        'portuguese')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_comment = ' '.join(tokens)
    return preprocessed_comment


def query(text):
    response = requests.post(API_URL, headers=headers, json={"text": text})
    return response.json()


@app.route('/analise', methods=['POST'])
def get_sentimento():
    data = request.get_json()
    if 'comentario' in data:
        comentario = data['comentario']
        comentario_preprocessado = preprocess_text(comentario)
        output = query(comentario_preprocessado)
        formatted_output = format_output(output)
        return jsonify({"resultado": formatted_output})
    else:
        return jsonify({"error": "O JSON deve conter uma chave 'comentario' com o texto a ser analisado."}), 400


def format_output(output):
    formatted_output = []
    for label_score in output:
        label = label_score['label']
        score = label_score['score'] * 100
        if score > 50:
            formatted_output.append(f"{label}: {score:.2f}%")
    return formatted_output


if __name__ == '__main__':
    # Running with Gunicorn, listening on 0.0.0.0 and using the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
