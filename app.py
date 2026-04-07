from flask import (
    Flask,
    render_template,
    request
)
from ml.predict import predict_image

app = Flask(__name__)


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict() -> str:
    file = request.files["image"]

    prediction, confidence = predict_image(file)

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)
