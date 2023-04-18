from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open("RFmodel.pkl", 'rb'))


app = Flask(__name__)

@app.route('/analysis')
def analysis():
    return render_template("report.html")



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method =="POST":
        gender = request.form['gender']
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        disease = int(request.form['disease'])
        married = request.form['married']
        work = request.form['work']
        residence = request.form['residence']
        glucose = float(request.form['glucose'])
        bmi = float(request.form['bmi'])
        smoking = request.form['smoking']

        input_values = [gender,age,hypertension,disease,married,work,residence,glucose,bmi,smoking]
        user_input = np.array(input_values,dtype=object).reshape(1,10)

        pred = model.predict(user_input)

        print("the prediction is ",pred)
        ans_y = "AI Predicting that you have chances to get brain stroke"
        ans_n = "AI Predicting that you don't have chances to get brain stroke"
        if pred == 1:
            pred=ans_y
        else:
            pred=ans_n
        return render_template("index.html",pred=pred)
    else:
        return render_template("index.html")






if __name__ == "__main__":
    app.run(debug=True)