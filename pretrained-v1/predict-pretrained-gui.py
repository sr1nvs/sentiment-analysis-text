import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTextEdit, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

# i am feeling hopeless
# i am happy today
# ambatukam


class SentimentAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Classifier")

        self.initUI()

        MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        config = AutoConfig.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    def initUI(self):
        self.text_edit = QTextEdit(self)
        self.text_edit.setPlaceholderText("Enter the text...")
        self.analyze_button = QPushButton("Analyze", self)
        self.analyze_button.clicked.connect(self.analyzeText)
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.analyze_button)
        layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def analyzeText(self):
        text = self.text_edit.toPlainText()
        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt')

        output = self.model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        result_text = ""
        for i in range(scores.shape[0]):
            label = self.model.config.id2label[ranking[i]]
            score = scores[ranking[i]]
            result_text += f"{label.capitalize()} - {np.round(float(score), 4)}\n"

        sentiment = self.model.config.id2label[ranking[0]].capitalize()
        result_text += f"\nThis text is {sentiment}."

        self.result_label.setText(result_text)

def main():
    app = QApplication(sys.argv)
    window = SentimentAnalysisGUI()
    window.setGeometry(100, 100, 600, 400)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()