from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Get absolute path of current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]
    data = vectorizer.transform([review])
    result = model.predict(data)[0]
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
