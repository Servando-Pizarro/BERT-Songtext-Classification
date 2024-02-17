# Exploring Genre and Success Classification through Song Lyrics using DistilBERT: A Fun NLP Venture

## Overview

In this project, we delve into the analysis of song lyrics using AI as an LLM-DistilBERT model, a lighter version of BERT that retains most of its performance while being more resource-efficient. Our goal was to leverage natural language processing (NLP) to understand and predict various attributes of songs solely based on their lyrics. The tasks we focused on were genre classification, predicting the success of a song based on view counts, and estimating the release year of tracks.

We faced numerous challenges such as dealing with the nuances of language in music, which often includes slang, metaphors, and cultural references. Ensuring the model could interpret these correctly was a significant part of our work.

## Project Structure

- `DistilBERT_Views_2Class.ipynb`: This notebook contains our approach to classifying songs as hits or flops based on view counts, a proxy for song success.
- `DistilBERT_Year.ipynb`: In this notebook, we predict the release year of songs, which involved understanding lyrical trends over time.
- `Lyrics_DataAnalysis.ipynb`: Here we conduct exploratory data analysis to understand our dataset's underlying patterns and distributions.
- `requirements.txt`: A list of dependencies required to run the notebooks and replicate our analysis.

## Usage

To replicate our analysis or apply our methods to a new set of song lyrics, run the provided Jupyter notebooks. Each notebook is well-documented, ensuring ease of use and understandability.

## Results

We achieved a genre classification accuracy of 65% and a success prediction accuracy of 79%. Our success prediction model was particularly challenging due to the subjective nature of music popularity. For release year prediction, we overcame the challenge of linguistic evolution over decades by implementing a model with an RMSE of 14.18, which we consider a success given the complexity of the task.

## Dashboard Application

A dashboard application was developed to showcase our models' capabilities. It uses the models to predict genre, success, and release year, offering a user-friendly interface for interaction with our system.

![grafik](https://github.com/Servando-Pizarro/BERT-Songtext-Classification/assets/105354134/5edc548d-1c2c-4c57-9eda-9b7418c22800)


## Citation

If you find our work helpful, please consider citing it:

```
@misc{song_lyrics_distilbert,
  title={Exploring Genre and Success Classification through Song Lyrics using DistilBERT},
  author={Pizarro, S. and Zimmermann, M. and Offermann, S. and Reither, F.},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/your-repo-name}}
}
```

## Contributions

Contributions to this project are welcome! You can contribute by forking the repository, opening issues for discussion, or submitting pull requests with enhancements or fixes.

## Acknowledgments

- Thanks to AAAI for the paper format guidelines.
- Appreciation to Kaggle for providing the Genius Dataset, which was the fundament for our analysis.
