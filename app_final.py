from flask import Flask, request, jsonify
import numpy as np
import re
import string
import tensorflow as tf
import tensorflow_datasets as tfds
import os

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'dcnn_model.h5')
encoder_path = os.path.join(os.path.dirname(__file__), 'tokenized_encoder')

# Load the DCNN model and encoder
ensemble_model = tf.keras.models.load_model(model_path, compile=False)
encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file(encoder_path)

counter=0

def clean_text(input_text):
    # Replace URLs in the text
    input_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', input_text)
    
    # Remove numbers
    input_text = re.sub(r'\d+', '', input_text)

    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation.replace('?', ''))
    input_text = input_text.translate(translator)

    return input_text

def tokenize_text(text):
    seq = encoder.encode(text)
    return seq

def checkForBullying(text):
    encoded_text = tokenize_text(text)
    output = ensemble_model(np.array([encoded_text]), training=False).numpy()
    labels = ['Not Controversial', 'Controversial']
    highest_index = np.argmax(output)
    predicted_label = labels[highest_index]

    # Calculate the sum of the output array
    sum_output = sum(output[0])

    for i in range(len(output[0])):
        output[0][i] = output[0][i] / sum_output * 100

    if predicted_label == "Controversial":
            global counter
            counter += 1
        
    print ("Counter: ",counter)

    return predicted_label, output

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.get_json().get('text')
        if not text:
            return jsonify({'error': 'Text is missing in the request.'}), 400

        cleaned_text = clean_text(text)
        if not cleaned_text:
            return jsonify({'error': 'Cleaned text is empty after preprocessing.'}), 400

        predicted_label, output = checkForBullying(cleaned_text)

        output = output.tolist()
        print (predicted_label)
        
        

        return jsonify({'label': predicted_label, 'output': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=4000)
