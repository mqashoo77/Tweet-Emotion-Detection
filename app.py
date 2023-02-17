

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score,classification_report
from sklearn.svm import SVC           # used 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,VotingClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB  # used
from sklearn.tree import DecisionTreeClassifier            # used
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import sys

import pickle
import aranorm as aranorm

from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)



models_folder = './models/'
models_folder = models_folder.rstrip('/')

vectorizer = pickle.load(open(f'{models_folder}/vectorizer.pkl', 'rb'))
mnb = pickle.load(open(f'{models_folder}/mnb.pkl', 'rb'))
dtc = pickle.load(open(f'{models_folder}/dtc', 'rb'))
svm = pickle.load(open(f'{models_folder}/svm.pkl', 'rb'))


##############################################################################################################

def predict(X, vectorizer, mnb):
    result = mnb.predict(vectorizer.transform(X))
    #return mnb.predict(vectorizer.transform(X))
    return result

##############################################################################################################

@app.route('/', methods=['GET', 'POST'])
def upload_file():
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
    
    return '''<!doctype html>
<head>
    <title>Tweet Emotion Detiction</title>
    <script>
    function myFunction()
    {
        // clear the output text box from the text
        output_text_box = document.getElementById('output_text');
        output_text_box.innerHTML = '';
       
        var elements = document.getElementsByClassName("formVal");
        var formData = new FormData(); 
        
        for(var i=0; i<elements.length; i++)
        {
            formData.append(elements[i].name, elements[i].value);
        }
        var xmlHttp = new XMLHttpRequest();
            xmlHttp.onreadystatechange = function()
            {
                if(xmlHttp.readyState == 4 && xmlHttp.status == 200)
                {
                    response = xmlHttp.responseText;
                    output_text_box = document.getElementById('output_text');
                    console.log(response);
                    output_text_box.innerHTML = response;
                }
            }
            xmlHttp.open("post", "/"); 
            xmlHttp.send(formData); 
    }
    </script>
</head>
<body>
    <h1>Tweet Emotion Detection</h1>
<form method=post enctype=multipart/form-data>

    <h3>Pleas Enter the tweet here</h3>
    <div class="tweet-container">
        <textarea id="input_text"class='formVal' rows="5" cols="50" type="text" name="input_text" placeholder="التغريدة"></textarea> <br>
        <input id="input-button"type="submit" value="submit" onclick="myFunction(); return false;">
    </div>
    <h3>The Result</h3>
  <textarea id="output_text" class='formVal' rows="5" cols="50" type="text" name="output_text" placeholder="المشاعر المتوقعة"></textarea>
</form>
<footer class="footer">
    <p class="footer-title">
        powered by
    </p>
    <p class="footer-title"><i class="fa-solid fa-copyright"></i>&nbsp<span>Mohammad Qashoo 2022</span></p>
</footer>

</body>

<style>
    * {
        font-family: "Poppins", sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        scroll-behavior: smooth;
        color:#fff;
    }
    body{
        margin-top: 100px;
        background-image: url("/twitter-background.jpg");
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    h1{
        padding-bottom: 30px;
    }
    .tweet-container{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    #input-button{
        background-color: #025bbb;
        border: 10px;
        color: white;
        margin: 4px 2px;
        cursor: pointer;
        height: 50px;
        width: 100px;
    }
    .footer {
  padding-top: 20px;
  padding-bottom: 20px;
  width: 100%;
  justify-content: center;
}

.footer-title {
  font-size: .8em;
  font-weight: 300;
  display: flex;
  justify-content: center;
  flex: 1;
}

.footer-title i {
padding-top: 3px;
  
}
    
    
</style>

</html>
    '''


if __name__ == '__main__':
    app.run(debug=True)