# Song Lyrics Analysis with DistilBERT

## Overview
This repository houses the code and models for an innovative approach to song lyrics analysis. Utilizing DistilBERT, we have developed methods for genre classification, success prediction based on view counts, and release year prediction. Our models achieve a genre classification accuracy of 65% and a success prediction accuracy of 79%.



## Project Structure
- `DistilBERT_Views_2Class.ipynb`: Jupyter notebook for view-based success classification.
- `DistilBERT_Year.ipynb`: Jupyter notebook for release year prediction.
- `Lyrics_DataAnalysis.ipynb`: Jupyter notebook for exploratory data analysis on song lyrics.
- `requirements.txt`: List of Python dependencies.

## Usage
Run the Jupyter notebooks to replicate our analysis or use the provided models for your own song lyrics dataset.

## Results
Our models effectively classify genres and predict song success with high accuracy. The best model for release year prediction is Support Vector Machines with an RMSE of 14.18.

## Dashboard Application
We've also built a dashboard application that utilizes our models to provide genre and success predictions, as well as release year estimations, from song lyrics. 

## Citation
If you find our work useful, please cite it as follows:
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
We welcome contributions and suggestions! Feel free to fork the repository, open issues, or submit pull requests.

## Acknowledgments
- AAAI for paper format guidelines.
- Kaggle for providing the Genius Dataset.

