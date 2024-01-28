# Importing libraries
import streamlit as st
import pandas as pd
import torch
from torch import nn
import joblib
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification, BertModel, BertTokenizer
from torch.nn.functional import softmax

# Custom Model Classes for the Genre Classification
class LyricsRegressor(nn.Module):
    def __init__(self, bert_model, hidden_size, dropout_rate):
        super(LyricsRegressor, self).__init__()
        self.bert = bert_model
        self.regressor = nn.Sequential(
            nn.Linear(768, hidden_size),  # DistilBertModel output size is 768
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.last_hidden_state[:, 0]
        return self.regressor(pooled_output)

# Set page configuration
st.set_page_config(page_title="Song Analysis App", page_icon="🎧", layout="wide")
 
label_mapping = {"pop": 0, "rock": 1,'rb': 2, "country": 3, "rap":4}



# Preparing adding the stored trained dictionary again to the base BERT Model
# Load the BERT model and tokenizer 

base_model_name = 'distilbert-base-uncased'
bert_model = BertModel.from_pretrained(base_model_name)
bert_tokenizer = BertTokenizer.from_pretrained(base_model_name)
tokenizer = DistilBertTokenizer.from_pretrained(base_model_name)
model_views = DistilBertForSequenceClassification.from_pretrained(base_model_name, num_labels=2)  # Make sure num_labels is consistent

# Load  fine-tuned BERT model as a dictionary
fine_tuned_model_dict = torch.load('equal16k_6epochs_trained_model_2classes_views.pth', map_location=torch.device('cpu'))

# Update the base model's state_dict with the fine-tuned model's state_dict
model_views.load_state_dict(fine_tuned_model_dict['model'])  # Load the 'model' key from the state_dict

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load your trained models
model_complete = torch.load('model_complete(1).pth', map_location=torch.device('cpu'))
model_year = joblib.load('svm_model.pth')

# Set models to evaluation mode
model_complete.eval()
model_views.eval()

def predict_genre(text, tokenizer, model):
    if not text.strip():
        return "Please enter song lyrics."

    # Prepare input
    encoded_text = tokenizer.encode_plus(
        text, add_special_tokens=True, max_length=512, truncation=True,
        padding='max_length', return_tensors='pt'
    )
    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']

    # Define label mappings
    label_mapping = {"pop": 0, "rock": 1, 'rb': 2, "country": 3, "rap": 4}

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        final_pred = preds.item()

    # Map the prediction to the corresponding genre
    label_to_genre = {idx: genre for genre, idx in label_mapping.items()}
    return label_to_genre[final_pred]



def predict_views(text, tokenizer, model_views):
    if not text.strip():
        return "Please enter song lyrics."

    # Prepare input
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']

    with torch.no_grad():
        # Get logits from the model
        outputs = model_views(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Convert logits to probabilities
        probabilities = softmax(logits, dim=1)
        
        # Get the predicted class (0 or 1) by finding the index of the max logit
        # predicted_class will be 0 for "fail" and 1 for "success"
        predicted_class = torch.argmax(probabilities, dim=1).numpy()[0]

        if predicted_class==0:
            return "fail"
        else: 
            return "success"

# Predict function to use the Year Modell
        

# Preparing the Year model with the Embeddings prior to calling the SVM Model

def get_bert_embeddings(text, bert_model, bert_tokenizer, max_len=300):
    # Tokenize the text
    tokens = bert_tokenizer.tokenize(bert_tokenizer.decode(bert_tokenizer.encode(text, add_special_tokens=True)))

    # Truncate or split the tokens to fit the specified maximum sequence length
    if len(tokens) > max_len:
        # Truncate the tokens
        tokens = tokens[:max_len]
    else:
        # No need to truncate, but if still longer than max_len, split into chunks
        tokens = [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]

    # Convert tokens to input IDs
    input_ids = [bert_tokenizer.convert_tokens_to_ids(chunk) for chunk in tokens]

    # Flatten the list 
    input_ids = [item for sublist in input_ids for item in sublist] if isinstance(input_ids[0], list) else input_ids

    # Create attention mask
    attention_mask = [1] * len(input_ids)

    # Convert to PyTorch tensors
    input_ids = torch.tensor([input_ids])
    attention_mask = torch.tensor([attention_mask])

    # Get BERT embeddings
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)

    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

def predict_year(text, model_year, bert_model, bert_tokenizer):
    # Generate BERT embeddings for the new song lyrics
    # Generate BERT embeddings for the sample text
    embeddings = get_bert_embeddings(text, bert_model, bert_tokenizer)

    #  Reshape the embeddings to 2D since sklearn requires 2D array for prediction
    embeddings_2d = embeddings.reshape(1, -1)

    # Use the model_year to predict the year
    predicted_year = model_year.predict(embeddings_2d)

    # Predict the year using the SVM model
    predicted_year = model_year.predict([embeddings])
    rounded_year = np.round(predicted_year).astype(int)  # Round to the nearest integer

    return rounded_year[0]

##############################

#App Design starts here


st.title('Song Analysis App with BERT')

st.markdown("""
This app uses BERT-based models to analyze song lyrics.
Enter the lyrics of a song to classify its genre, predict its release year, and estimate its number of hearings!
""")

# Sidebar
st.sidebar.header("About the App")
st.sidebar.info("This Streamlit app is designed to analyze song lyrics using BERT-based models.")

# Text input
user_input = st.text_area("Enter song lyrics here:", "Type your lyrics...")

if st.button('Analyze Song'):
    with st.spinner('Analyzing...'):
        genre_result = predict_genre(user_input, tokenizer, model_complete)
        year_result = predict_year(user_input, model_year, bert_model, bert_tokenizer)  # Pass the BERT model and tokenizer
        views_result = predict_views(user_input, tokenizer, model_views)

        st.success(f'Genre Prediction: {genre_result}')
        st.success(f'Predicted Release Year: {year_result}')
        st.success(f'It is a: {views_result}')

# Footer
st.markdown("---")
st.info("Developed by Servando, Flo, Serkan and Moritz")

# Run app with the following command in the terminal:
# streamlit run Dashboard_app.py



