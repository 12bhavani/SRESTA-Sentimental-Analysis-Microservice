import pandas as pd  # to load dataset
import numpy as np  # for mathematic equation
from sklearn.preprocessing import LabelEncoder
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from keras.models import Sequential
from keras.layers.core import Dense
import re
from keras.models import load_model

# load the training examples
data = pd.read_csv(r'C:\Users\DELL\Downloads\setit\training.csv')
print(data)

le = LabelEncoder()
y = data['label'].values
# sw = set(stopwords.words('english'))
# ps = PorterStemmer()
tfidf = TfidfTransformer()
cv = CountVectorizer()


# defining a method for preprocessing data
def cleaning_text(sample):
    sample = sample.lower()
    sample = sample.replace("<br /><br >", "")
    sample = re.sub("[^a-zA-Z]+", " ", sample)
    sample = sample.split()
    # sample = [ps.stem(s) for s in sample if s not in sw]
    sample = " ".join(sample)
    return sample


# preprocessing the training examples
data['clean_text'] = data['text'].apply(cleaning_text)
print(data['clean_text'])
corpus = data['clean_text'].values
X = cv.fit_transform(corpus)
X = tfidf.fit_transform(X)
# creating an object for sequential model for ANN construction
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(X.shape[1],)))
model.add(Dense(128, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(6, activation="softmax"))
model.summary()
# model compilation
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# splitting into validation and training sets
X_val = X[:1600]
X_train = X[1600:]
y_val = y[:1600]
y_train = y[1600:]
# sorting the X data using sorted_indices
X_train = X_train.sorted_indices()
X_val = X_val.sorted_indices()
# inscribing the model into fit method
history = model.fit(X_train, y_train, epochs=20, batch_size=64, shuffle=True, validation_data=(X_val, y_val))
# save the model
model.save('saved_model.h5')
# load the saved model
loaded_model = load_model('saved_model.h5')

# testing part of the model
# load the test-data
data = pd.read_csv(r'C:\Users\DELL\Downloads\setit\test12.csv')
# storing the test data into a test variable
test = pd.read_csv('test12.csv')
test.shape
test.head()
test['clean_text'] = test['text'].apply(cleaning_text)
X_test = test['clean_text']
X_test = cv.transform(X_test)
X_test = tfidf.transform(X_test)
X_test = X_test.sorted_indices()
y_pred = model.predict(X_test)
y_pred
y_test = test['label']
# Convert SparseTensor to dense numpy array
X_test = X_test.toarray()
# Evaluate model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
# Print test accuracy
print('Test accuracy:', test_acc)
print('loss : ', test_loss)


# predict method which works on the saved model
def predict(inpt):
    c_l = cleaning_text(inpt)
    arr = np.array([c_l])
    arr = cv.transform(arr)
    arr = tfidf.transform(arr)
    arr = arr.sorted_indices()
    prediction = model.predict([arr])
    predicted_class = np.argmax(prediction)
    print(predicted_class)
    # depending on the output the sentiment is driven
    if predicted_class == 0:
        display = "sadness"
    elif predicted_class == 1:
        display = "joy"
    elif predicted_class == 2:
        display = "love"
    elif predicted_class == 3:
        display = "angry"
    elif predicted_class == 4:
        display = "fear"
    elif predicted_class == 5:
        display = "surprise"
    else:
        display = "calm"
    return display
