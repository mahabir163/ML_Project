from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,predictPipeline


application = Flask(__name__) #Entry point

app = application

#Route for a home page
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            price=request.form.get("price"),
            p1h=request.form.get("1h"),
            p24h=request.form.get("24h"),
            p7d=request.form.get("7d"),
            p24h_volume=request.form.get("24h_volume")
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = predictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("home.html",results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)