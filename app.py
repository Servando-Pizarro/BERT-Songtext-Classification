import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# Set page configuration with lion icon
st.set_page_config(page_title="Song Genre Classifier", page_icon="🦁", layout="wide")

# Load dataset and create label mapping
df = pd.read_csv('sampled_english_songs_10000.csv')
label_mapping = {label: idx for idx, label in enumerate(df['tag'].unique())}

# Change model_name to 'albert-base-v2'
model_name = 'albert-base-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load your trained model
# Make sure to change the model architecture here as well
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))
model.load_state_dict(torch.load('llm_model.pth', map_location=torch.device('cpu')))
model.eval()

def predict(text, tokenizer, model):
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted_label_idx = torch.max(outputs.logits, dim=1)

    label_to_genre = {idx: genre for genre, idx in label_mapping.items()}
    predicted_genre = label_to_genre[predicted_label_idx.item()]

    return predicted_genre


st.title('Song Genre Classifier with BERT')

st.markdown("""
This app uses a BERT-based model to classify song lyrics into different genres. 
Enter the lyrics of a song and see which genre the AI thinks it belongs to!
""")

# Sidebar
st.sidebar.header("About the App")
st.sidebar.info("This Streamlit app is designed to classify song lyrics into genres using a BERT-based model. It's a demonstration of natural language processing capabilities.")

# Text input
user_input = st.text_area("Enter song lyrics here:", "Type your lyrics...")

if st.button('Predict Genre'):
    with st.spinner('Analyzing...'):
        result = predict(user_input, tokenizer, model)
        st.success(f'Prediction: {result}')

# Footer
st.markdown("---")
st.info("Developed by Servando, Flo, Serko_7, Moritz")

# To run it do: py -m streamlit run "C:\\Users\\serva\\OneDrive\\Dokumente\\GitHub\\Project Songs ADA\\app.py"