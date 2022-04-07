from cgi import test
# from crypt import methods
from pyexpat import features
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__) 
# template_folder='template'
model = pickle.load(open('model/model_houseprices.pkl', 'rb'))

# @app.route('/hello')
# @app.route('/hello/<name>')
# def hello_world(name=""):
#     if (name):
#         return "Hello " + str(name)
#     else:
#        return 'Hello, PYTN 70!'

@app.route('/')
def main():
    return (render_template('main.html'))

@app.route("/predict", methods=['POST'])
def predict():
    features = []
    for val in request.form.values():
        features.append(val)

    house = int(features[0])
    house1 = int(features[1])
    house2 = int(features[2])

    # label_prediksi = ["main", "tidak main"]
    hasil_prediksi = model.predict([[house, house1, house2]])[0]

    # return str(features
    
    return render_template("main.html", house=house, hasil=hasil_prediksi)#[ind])

    # print(request.form.values())
    # return str(request.form.values())
    #  render_template























# @app.route('/predict', methods=["POST"])
# def predict_placement():
    # terima input dari file html, kita ubah format jadi np.array 2D, 4 kolom untuk 4 fitur
    # supaya bisa digunakan di model machine learning kita
#     features = []
#     for x in request.form.values():
#         features.append(int(x))
    # ubah jadi 2D
#     fitur = [features]
    # ubah jadi numpy array
#     fitur = np.array(features).reshape((1,4))
    
#     hasil_prediksi = model.predict(features) #diterima/tidak
    
    # format hasil text yang ditampilkan
#     output = {"Not Placed": "tidak diterima", "Placed": "diterima"}
#     hasil = output[hasil_prediksi[0]]
    
    # hasil prediksi dikirim balik ke html untuk di display ke user    
#     return render_template('main.html', prediction_text='Student sepertinya akan {} ke tempat kerja. Nilai student adalah sebagai berikut: smp {}, sma {}, S1 {}, tes masuk {}'.format(hasil, features[0,0], features[0,1], features[0,2], features[0,3]))