import re
from joblib.parallel import method
import nltk
nltk.download('stopwords')  
from nltk.corpus import stopwords
import pickle
from flask import Flask, render_template,request

def clean_text(text):
    text = re.sub(r'[^A-Za-z\s]', '',text)  #Remove special charaters and numbers
    text = text.lower() #converting to lower case
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stop words
    return text

app = Flask(__name__)
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('SentimentModel.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name']
        review = request.form['review']
        clean_txt = clean_text(review)
        vector_txt = vectorizer.transform([clean_txt])
        prediction = model.predict(vector_txt)
        return render_template('index.html', name=name, review=review, result=prediction[0])
    except Exception as e:
        return f"Error: {e}", 400

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # fallback to 10000 if PORT not set
    app.run(host='0.0.0.0', port=port, debug=False)
