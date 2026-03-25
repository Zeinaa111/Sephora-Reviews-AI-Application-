import gradio as gr
import pickle

model = pickle.load(open("../model/model.pkl", "rb"))
vectorizer = pickle.load(open("../model/vectorizer.pkl", "rb"))

def predict_sentiment(review):
    clean = review.lower()
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)[0]
    return prediction

interface = gr.Interface(
    fn=predict_sentiment,
    inputs="text",
    outputs="text",
    title="Sephora Sentiment Analysis ",
    description="Enter a review to analyze sentiment"
)

interface.launch()


