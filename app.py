import streamlit as st 
import numpy as np
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


#load the model and tokenizer
model=load_model("next_word_prediction_model.keras")
with open("tokenizer.pickle","rb")as handle:
    tokennize=pickle.load(handle)

#function to predict next word
def predict_next_word(model,tokennize,text,max_sequence_len):
    sequence=tokennize.texts_to_sequences([text])[0]
    padded_sequence=pad_sequences([sequence],maxlen=max_sequence_len-1,padding="pre")
    predicted_probs=model.predict(padded_sequence,verbose=0)
    predicted_index=np.argmax(predicted_probs,axis=-1)[0]
    for word,index in tokennize.word_index.items():
        if index==predicted_index:
            return word
    return None 

#streamli app
st.title("Next Word Prediction using LSTM RNN")

input_text=st.text_input("Enter a partial sentence:")

if st.button("Predict Next Word"):
    max_sequence_len=model.input_shape[1]+1
    next_word=predict_next_word(model,tokennize,input_text,max_sequence_len)
    st.write(f"Predicted Next Word: {next_word}")