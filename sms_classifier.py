import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string



# Defining a function 'transform_text' to preprocess text data
def transform_text(text):
    # Creating a PorterStemmer instance for stemming words
    ps = PorterStemmer()

    # Converting text to lowercase
    text = text.lower()

    # Tokenizing the text into words
    text = nltk.word_tokenize(text)

    # Filtering out non-alphanumeric characters
    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()

    # Removing stopwords and punctuation from the text
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    # Applying stemming to each word in the text
    for i in text:
        y.append(ps.stem(i))

    # Joining the processed words to form a cleaned text
    return " ".join(y)

tfidf = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))





st.title("Email/SMS spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # 1 preprocess
    transformed_sms = transform_text(input_sms)
    # 2 vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3 pridict
    result = model.predict(vector_input)[0]
    # Display

    if result == 1:
        st.header("Spam")
    else:
        st.header("not Spam")
