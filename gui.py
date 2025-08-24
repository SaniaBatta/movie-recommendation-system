import tkinter as tk
from tkinter import ttk
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading the data from the csv file to a pandas dataframe
movies_data = pd.read_csv('movies (1).csv')

# selecting the relevant features for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# replacing the null values with null string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# combining all the selected features
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# converting the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# calculating similarity matrix
similarity = cosine_similarity(feature_vectors)

# Function to recommend similar movies
def recommend_similar_movies():
    movie_name = movie_entry.get()
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        recommendation_text.set('Movies suggested for you:')
        i = 1
        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = movies_data[movies_data.index == index]['title'].values[0]
            if i < 30:
                recommendation_text.set(recommendation_text.get() + f"\n{i}. {title_from_index}")
                i += 1
    else:
        recommendation_text.set("Movie not found in the database.")

# Creating GUI
root = tk.Tk()
root.title("Movie Recommendation System")

# Movie Name Label and Entry
movie_label = ttk.Label(root, text="Enter your favorite movie name:")
movie_label.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)

movie_entry = ttk.Entry(root)
movie_entry.grid(row=0, column=1, padx=10, pady=5)

# Recommend Button
recommend_button = ttk.Button(root, text="Recommend", command=recommend_similar_movies)
recommend_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Recommendation Text
recommendation_text = tk.StringVar()
recommendation_label = ttk.Label(root, textvariable=recommendation_text, wraplength=400)
recommendation_label.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

root.mainloop()
