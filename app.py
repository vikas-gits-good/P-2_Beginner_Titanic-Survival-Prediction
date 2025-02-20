from flask import Flask, request, render_template

from src.logger_predict import logging
from src.pipeline.pipeline_prediction import CustomData, PredictionPipeline

application = Flask(__name__)
app = application


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        logging.info("Getting user input from web form")
        data = CustomData(
            Pclass=request.form.get("Pclass"),
            Sex=request.form.get("Sex"),
            Age=request.form.get("Age"),
            SibSp=request.form.get("SibSp"),
            Parch=request.form.get("Parch"),
            Fare=request.form.get("Fare"),
            Cabin=request.form.get("Cabin"),
            Embarked=request.form.get("Embarked"),
        )
        x_pred = data.get_DataFrame()

        logging.info("Calculating prediction for user input")
        ppln_pred = PredictionPipeline()
        y_pred = ppln_pred.predict(features=x_pred)
        surv = "Survived" if y_pred == 1 else "Died"
        logging.info("Returning prediction to user")
        return render_template("home.html", results=surv)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
