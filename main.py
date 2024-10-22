from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo import MongoClient

# Initialize FastAPI app
app = FastAPI()

# MongoDB connection
client = MongoClient("mongodb+srv://anup:1234@cluster0.kuk95.mongodb.net/")
db = client["Tutor"]  # Replace with your database name
collection = db["teachers"]  # Replace with your collection name

# Preprocessing the expertise field and handling missing data gracefully
def preprocess_expertise_field(row):
    # Check if 'expertise' is a list and join the elements; if it's not, return an empty string
    if isinstance(row.get('expertise'), list):
        return " ".join(row['expertise']).lower()
    else:
        return str(row.get('expertise', '')).lower()

# Fetch teachers data from MongoDB
def load_teachers_from_db():
    teachers_data = list(collection.find({}, {'_id': 0, 'bio': 1, 'expertise': 1, 'contactInfo': 1, 'education': 1, 'profileImage': 1}))
    
    # Convert the list of documents to DataFrame
    teachers_df = pd.DataFrame(teachers_data)
    
    # Preprocess the expertise field
    teachers_df['subjects'] = teachers_df.apply(preprocess_expertise_field, axis=1)
    
    # Handle missing nested fields
    teachers_df['contactInfo.address'] = teachers_df.apply(lambda row: row.get('contactInfo', {}).get('address', ''), axis=1)
    teachers_df['education[0].school'] = teachers_df.apply(lambda row: row.get('education', [{}])[0].get('school', ''), axis=1)
    
    # Selecting the relevant columns
    teachers_df = teachers_df[['bio', 'subjects', 'contactInfo.address', 'education[0].school', 'profileImage']]
    teachers_df.fillna('', inplace=True)
    
    return teachers_df

# Load data and prepare TF-IDF matrix
teachers_df = load_teachers_from_db()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(teachers_df['subjects'])

# Function to recommend teachers based on subject
def recommend_teachers(subject, top_n=5):
    subject = subject.lower()
    subject_tfidf = tfidf.transform([subject])
    similarity_scores = cosine_similarity(subject_tfidf, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    if similarity_scores.max() == 0:
        return None  # No relevant teacher found
    
    return teachers_df.iloc[top_indices]

# FastAPI route to get teacher recommendations
@app.get("/recommend/")
def get_recommendations(subject: str, top_n: int = 5):
    recommendations = recommend_teachers(subject, top_n)
    
    if recommendations is None or recommendations.empty:
        raise HTTPException(status_code=404, detail="No teacher found for the given subject")
    
    response = recommendations.to_dict(orient='records')
    return {"recommendations": response}

# Add a route for the root path
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}
