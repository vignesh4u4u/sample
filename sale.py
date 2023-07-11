from flask import Flask,request,render_template
import pickle as pk
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.preprocessing import  StandardScaler,LabelEncoder
import numpy
import joblib
app = Flask(__name__,template_folder="template")
read = pk.load(open("sales.pkl", "rb"))
read = joblib.load("sales.pkl")
joblib.dump(read, "sales.pkl")
@app.route("/")
@app.route("/vicky")
def add():
    return render_template("sales.html")
@app.route("/predict",methods=["POST","GET"])
def pre():
    if request.form=="POST":
        enc = LabelEncoder()
        source = request.form["Source"]
        sales_Agent = request.form["Sales_Agent"]
        location = request.form["Location"]
        delivery_Mode = request.form["Delivery_Mode"]
        year = int(request.form["year"])
        month = int(request.form["month"])
        Source = enc.fit_transform(source)
        Sales_Agent = enc.fit_transform(sales_Agent)
        Location = enc.fit_transform(location)
        Delivery_Mode = enc.fit_transform(delivery_Mode)
        result = read.predict([[Source,Sales_Agent,Location,Location,Delivery_Mode,year,month]])[0]
        return render_template("sales.html",**locals())
if __name__=="__main__":
    app.run(debug=True,port=5000,host='0.0.0.0')