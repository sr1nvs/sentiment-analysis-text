import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import json
import numpy as np
from PyQt5.QtCore import Qt

class EmotionClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the trained model
        self.model = load_model("emotion_classification_model_v2.h5")

        # Load the tokenizer's word index
        with open("tokenizer_word_index.json", "r") as json_file:
            tokenizer_word_index = json.load(json_file)

        # Recreate the tokenizer using the loaded word index
        self.tokenizer = Tokenizer()
        self.tokenizer.word_index = tokenizer_word_index

        self.max_sequence_length = 100

        self.emotion_labels = {
            0: "Negative",
            1: "Neutral",
            2: "Positive",
        }

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Dollar Store Emotion Classifier")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        label = QLabel("Enter your text:")
        layout.addWidget(label)

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter text here...")
        layout.addWidget(self.input_text)

        self.predict_button = QPushButton("Predict Emotion")
        self.predict_button.clicked.connect(self.predict_and_display)
        layout.addWidget(self.predict_button)

        self.output_label = QLabel()
        layout.addWidget(self.output_label)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def predict_emotion(self, user_input):
        user_input_sequence = self.tokenizer.texts_to_sequences([user_input])
        user_input_sequence = pad_sequences(user_input_sequence, maxlen=self.max_sequence_length)
        predicted_class_probs = self.model.predict(user_input_sequence)[0]
        predicted_class = np.argmax(predicted_class_probs)
        return predicted_class, self.emotion_labels[predicted_class], predicted_class_probs

    def predict_and_display(self):
        user_input = self.input_text.toPlainText()
        if user_input:
            predicted_class, emotion_label, predicted_class_probs = self.predict_emotion(user_input)
            
            confidence_values = "\n".join([f"{self.emotion_labels[i]}: {prob:.4f}" for i, prob in enumerate(predicted_class_probs)])
            
            self.output_label.setText(f"Confidence values:\n{confidence_values}\n\nPredicted class: {predicted_class} - {emotion_label}")
            self.output_label.setAlignment(Qt.AlignCenter)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionClassifierApp()
    window.show()
    sys.exit(app.exec_())
