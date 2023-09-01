from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import json
import numpy as np

# Load the trained model
model = load_model("emotion_classification_model.h5")

# Load the tokenizer's word index
with open("tokenizer_word_index.json", "r") as json_file:
    tokenizer_word_index = json.load(json_file)

# Recreate the tokenizer using the loaded word index
tokenizer = Tokenizer()
tokenizer.word_index = tokenizer_word_index

max_sequence_length = 100 

# Define a dictionary to map class indices to labels
emotion_labels = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
}
def predict_emotion(user_input):
    # Preprocess the user input using the same tokenizer and padding as in training
    user_input_sequence = tokenizer.texts_to_sequences([user_input])
    user_input_sequence = pad_sequences(user_input_sequence, maxlen=max_sequence_length)

    # Make a prediction using the model
    predicted_class_probs = model.predict(user_input_sequence)[0]
    predicted_class = np.argmax(predicted_class_probs)

    return predicted_class, emotion_labels[predicted_class]

if __name__ == "__main__":
    user_input = input("Enter text : ")
    predicted_class, emotion_label = predict_emotion(user_input)
    print(f"Predicted class: {predicted_class} - {emotion_label}")
