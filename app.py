from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Model prediction will be added here later
    return "Apple classification model coming soon!"


if __name__ == "__main__":
    app.run(debug=True)
