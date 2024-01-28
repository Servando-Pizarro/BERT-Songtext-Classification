# Song Analysis App with BERT

This repository contains the code for a Streamlit dashboard app that leverages BERT-based models to analyze song lyrics. The application classifies songs by genre, predicts their release year, and estimates the number of views using the lyrics.

## Project Structure

- `Dashboard_app.py`: The main Streamlit application for the song analysis dashboard.
- `DistilBERT_Views_2Class.ipynb`: A Jupyter notebook detailing the training process for the view classification model.
- `DistilBERT_Year.ipynb`: A Jupyter notebook for training the model to predict the release year of songs.
- `Lyrics_DataAnalysis.ipynb`: Data analysis Jupyter notebook for the lyrics dataset.
- `requirements.txt`: A list of Python dependencies required to run the app.
- `model_complete(1).pth`: The trained model file for song genre classification.
- `model_years.pth`: The trained model for predicting the release year based on BERT embeddings.
- `svm_model.pth`: The SVM model file for year prediction.
- `.gitattributes` & `.gitignore`: Git configuration files.

## Installation

To get the application running on your system, follow these steps:

1. Clone this repository to your local machine.
2. Make sure Python is installed on your system.
3. Install the necessary Python packages using the following command:

    ```bash
    pip install -r requirements.txt
    ```

4. The models should be trained using the provided Jupyter notebooks if you wish to retrain or fine-tune them. Make sure you have the required dataset.

5. To start the Streamlit dashboard, run this command in your terminal:

    ```bash
    streamlit run Dashboard_app.py
    ```

## Usage

When you launch the app, you will see a text input area where you can paste the lyrics of a song. After entering the lyrics, click the "Analyze Song" button to receive predictions for the genre and release year, as well as a classification of the number of views.

## Models

The models need to be trained before using the dashboard app, or you can integrate your own pre-trained language models.

- For genre classification: Use `model_complete(1).pth`
- For release year prediction: Use `model_years.pth` for BERT embeddings and `svm_model.pth` for the SVM model.

Ensure that you have the trained model files in the correct directory before running the app.

## Contributions

Contributions to this project are welcome. You can fork the repository, make changes, and create a pull request with your improvements.

## Developers

- Servando
- Flo
- Serkan
- Moritz

Thank you for your interest in our song analysis project. We hope you find this application insightful and useful for your analysis needs!
