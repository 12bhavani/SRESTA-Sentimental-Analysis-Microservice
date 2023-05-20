from flask import Flask, render_template, request, redirect
from model import predict
from tensorflow import keras

# object creation for Flask
app = Flask(__name__)

# Load your sentiment analysis model
model = keras.models.load_model('saved_model.h5')


# route to index page
@app.route('/')
def index():
    return render_template('index.html')


# route to result page followed by redirecting to web page
@app.route('/predict', methods=['POST'])
def result():
    sentence = request.form.get('sentence')
    print(sentence)
    emotion = predict(sentence)
    print(emotion)
    return render_template('result.html', emotion=emotion)


# calling run method for flask framework
if __name__ == '__main__':
    app.run()
