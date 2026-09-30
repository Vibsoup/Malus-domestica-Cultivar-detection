from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        image = request.files.get("image")

        if image:
            prediction = "Prediction will be added soon."

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
