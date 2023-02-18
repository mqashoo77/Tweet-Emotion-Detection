
import numpy as np
import pickle
import aranorm as aranorm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC

from flask import Flask, flash, request, redirect, url_for,render_template

from werkzeug.utils import secure_filename



app = Flask(__name__)

models_folder = './models/'
models_folder = models_folder.rstrip('/')

vectorizer = pickle.load(open(f'{models_folder}/vectorizer.pkl', 'rb'))
mnb = pickle.load(open(f'{models_folder}/mnb.pkl', 'rb'))
dtc = pickle.load(open(f'{models_folder}/dtc.pkl', 'rb'))
svm = pickle.load(open(f'{models_folder}/svm.pkl', 'rb'))



def predict( tweet, vectorizer, mnb):
    result = mnb.predict(vectorizer.transform(tweet))
    #return mnb.predict(vectorizer.transform(X))
    return result

@app.route('/',methods = ['POST', 'GET'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'input_text' not in request.form:
            flash('No text found!')
            return redirect(request.url)

        text = request.form['input_text']
        text = aranorm.normalize_arabic_text(text)

        if text == '':
            return 'Please, write an Arabic sentance. Symbols and non-Arabic characters will be removed from the text....'

        predcited = predict(np.array([text]), vectorizer, mnb)
        predcited = str(predcited.squeeze())

        print(f'text: {text}')
        print("Predicted: ", predcited)
        return predcited

    return render_template("index.html")

app.run()