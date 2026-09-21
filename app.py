import streamlit as st
import joblib

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


@st.cache_resource
def load_model():

    movies = joblib.load("movies.pkl")
    cosine_sim = joblib.load("cosine_similarity.pkl")
    indices = joblib.load("movie_indices.pkl")

    return movies, cosine_sim, indices


movies, cosine_sim, indices = load_model()


def recommend_movies(title, num_recommendations):

    if title not in indices:
        return []

    idx = indices[title]

    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[
        1:num_recommendations + 1
    ]

    recommendations = []

    for movie_index, score in similarity_scores:

        recommendations.append({
            "title": movies.iloc[movie_index]["title"],
            "genres": movies.iloc[movie_index]["genres"],
            "similarity": float(score)
        })

    return recommendations


st.title("Movie Recommendation System")

st.write(
    """
    This recommendation system uses **Content-Based Filtering**
    to recommend movies based on genre similarity.
    """
)

st.divider()


st.sidebar.header("Recommendation Settings")

num_recommendations = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)


movie_list = sorted(
    movies["title"].tolist()
)

selected_movie = st.selectbox(
    "Select a movie",
    movie_list
)



if st.button("Get Recommendations"):

    recommendations = recommend_movies(
        selected_movie,
        num_recommendations
    )

    st.subheader(
        f"Recommendations based on: {selected_movie}"
    )

    if recommendations:

        for i, movie in enumerate(
            recommendations,
            start=1
        ):

            st.markdown(
                f"""
                ### {i}. {movie['title']}

                **Genres:** {movie['genres']}

                **Similarity Score:** {movie['similarity']:.2f}

                ---
                """
            )

    else:

        st.warning(
            "Sorry, no recommendations were found."
        )


st.sidebar.divider()

st.sidebar.info(
    """
    **Project:** Movie Recommendation System

    **Method:** Content-Based Filtering

    **Algorithm:** TF-IDF + Cosine Similarity

    **Dataset:** MovieLens
    """
)
