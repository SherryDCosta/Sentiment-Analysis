import re
from joblib.parallel import method
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
model = pickle.load(open('sentiment_model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        name = request.form.get('Name')
        review = request.form.get('Review')
        clean_txt = clean_text(review)
        vector_txt = vectorizer.transform([clean_txt])
        prediction =  model.predict(vector_txt)
        return render_template('index.html',name = name, review = review, text = f'{prediction[0]} Sentiment')
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)