# sentiment-analysis-text

sentiment analysis models based on text input
consists of two models - one pretrained ([roBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)) and the other model which i trained myself based on some random dataset i found which explains its skill issue.
also has a gui based predictor (my first time using pyqt5)

# usage

run predict-gui.py to use my model
and pretrained-v1/predict-pretrained-gui.py to use the already trained model

my model classifies the text into one of six classes while the pretrained model can only
classify as postive/neutral/negative
