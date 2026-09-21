# Movie Recommendation System

## Overview

This project is a content-based movie recommendation system
developed as part of the EncoderX AI/ML Internship - Week 03.

The system recommends movies based on the similarity between
their genres.

## Objective

The objective is to build an intelligent recommendation engine
that can suggest relevant movies based on a movie selected by
the user.

## Dataset

The project uses the MovieLens dataset.

The main file used is:

movies.csv

Important columns:

- movieId
- title
- genres

## Data Preprocessing

The following preprocessing steps were performed:

- Missing value checking
- Duplicate removal
- Irrelevant record removal
- Genre text preprocessing
- Index resetting

## Recommendation Approach

Content-Based Filtering was used.

The recommendation pipeline is:

Movie Genres
↓
TF-IDF Vectorization
↓
Cosine Similarity
↓
Top-K Recommendations

## Evaluation

The system was evaluated using:

- Precision@10
- Recall@10

The actual evaluation scores are reported in the project
notebook.

## Interface

A Streamlit interface was developed that allows users to:

- Select a movie
- Choose the number of recommendations
- View recommended movies
- View similarity scores

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Google Colab

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
