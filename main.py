import sys
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from NLP import NLPProcessor
from ReteaNeuronala import Perceptron
import json

def read_input_text():
    parser = argparse.ArgumentParser(description='Process text about a cat.')
    parser.add_argument('--file', type=str, help='Path to the input file')
    parser.add_argument('--text', type=str, help='Input text about a cat')
    args = parser.parse_args()

    if args.file:
        with open(args.file, 'r') as file:
            text = file.read()
    elif args.text:
        text = args.text
    else:
        raise ValueError("Either --file or --text must be provided.")
    
    return text

def process_text(text):
    processor = NLPProcessor(text)
    keywords = processor.extract_keywords()
    return keywords

def predict_breed(keywords, model_path='model.pkl'):
    # Load the trained model
    perceptron = Perceptron.load_model(model_path)

    keyword_vector = np.zeros(len(perceptron.train_x[0]))
    for keyword in keywords:
        if keyword in perceptron.feature_names:
            keyword_vector[perceptron.feature_names.index(keyword)] = 1
    
    # Predict the cat breed
    prediction = perceptron.predict(np.array([keyword_vector]), perceptron.W1, perceptron.b1, perceptron.W2, perceptron.b2)
    predicted_label_index = np.argmax(prediction, axis=1)[0]
    
    with open('race_codification.json', 'r') as json_file:
        race_codification = json.load(json_file)
    
    # Map the predicted label index to the breed name
    predicted_breed = race_codification[str(predicted_label_index)]
    
    return predicted_breed

def generate_description(text, predicted_label):
    # Generate a small description based on the input text and the predicted cat breed
    description = f"The input text describes a cat that is predicted to be of breed {predicted_label}."
    return description

if __name__ == "__main__":
    text = read_input_text()
    keywords = process_text(text)
    predicted_label = predict_breed(keywords)
    description = generate_description(text, predicted_label)
    
    print(f"Predicted Label: {predicted_label}")
    print(description)