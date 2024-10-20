from fastapi import FastAPI # type: ignore
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize FastAPI app
app = FastAPI()

# Load the CSV file
file_path = 'Tutor.teachers.csv'
teachers_df = pd.read_csv(file_path)

# Preprocessing
teachers_df['subjects'] = teachers_df['expertise[0]'].fillna('') + " " + teachers_df['expertise[1]'].fillna('')
teachers_df = teachers_df[['bio', 'subjects', 'contactInfo.address', 'education[0].school', 'profileImage']]
teachers_df.fillna('', inplace=True)
teachers_df['subjects'] = teachers_df['subjects'].str.lower()

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(teachers_df['subjects'])

# Function to recommend teachers based on subject
def recommend_teachers(subject, top_n=5):
    subject = subject.lower()
    subject_tfidf = tfidf.transform([subject])
    similarity_scores = cosine_similarity(subject_tfidf, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    return teachers_df.iloc[top_indices]

# FastAPI Route to get teacher recommendations
@app.get("/recommend/")
def get_recommendations(subject: str, top_n: int = 5):
    recommendations = recommend_teachers(subject, top_n)
    response = recommendations.to_dict(orient='records')
    return {"recommendations": response}



# Add a route for the root path
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}

