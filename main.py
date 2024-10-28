from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from bson import ObjectId
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import datetime
import logging

# Initialize FastAPI app
app = FastAPI()

# MongoDB connection
client = MongoClient("mongodb+srv://anup:1234@cluster0.kuk95.mongodb.net/")
db = client["Tutor"]
teacher_collection = db["teachers"]
user_collection = db["users"]

# Ensure 2dsphere index on the location field
teacher_collection.create_index([("location", "2dsphere")])

# Set up logging
logging.basicConfig(level=logging.INFO)

# Helper function to convert ObjectId and other non-serializable fields
def serialize_doc(doc):
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            doc[key] = str(value)
        elif isinstance(value, datetime.datetime):
            doc[key] = value.isoformat()
        elif isinstance(value, dict):
            doc[key] = serialize_doc(value)
        elif isinstance(value, list):
            doc[key] = [serialize_doc(item) if isinstance(item, dict) else item for item in value]
    return doc

# Helper function to fetch user info (name and email) from user collection by teacher's ID
def get_user_info(teacher_id):
    user_data = user_collection.find_one({"_id": ObjectId(teacher_id)}, {"fullName": 1, "email": 1})
    if user_data:
        return {"name": user_data.get("fullName", "Unknown"), "email": user_data.get("email", "")}
    return {"name": "Unknown", "email": ""}

# Preprocess the expertise field and handle missing data gracefully
def preprocess_expertise_field(row):
    if isinstance(row.get('expertise'), list):
        return " ".join(row['expertise']).lower()
    else:
        return str(row.get('expertise', '')).lower()

# Fetch teachers data from MongoDB
def load_teachers_from_db():
    teachers_data = list(teacher_collection.find({}, {'_id': 1, 'bio': 1, 'expertise': 1, 'contactInfo': 1, 'education': 1, 'profileImage': 1, 'coverImage': 1, 'status': 1, 'createdAt': 1, 'updatedAt': 1, 'messages': 1, 'notifications': 1}))
    teachers_data = [serialize_doc(doc) for doc in teachers_data]
    teachers_df = pd.DataFrame(teachers_data)
    teachers_df['subjects'] = teachers_df.apply(preprocess_expertise_field, axis=1)
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
    filtered_indices = [i for i in top_indices if similarity_scores[i] > 0]

    if len(filtered_indices) == 0:
        return None

    recommended_teachers = teachers_df.iloc[filtered_indices].copy()
    recommended_teachers = recommended_teachers.drop(columns=['subjects'])
    return recommended_teachers

# Route for recommending teachers based on subject
@app.get("/recommend/")
async def get_recommendations(subject: str, top_n: int = 5):
    recommendations = recommend_teachers(subject, top_n)
    if recommendations is None or recommendations.empty:
        raise HTTPException(status_code=404, detail="No teacher found for the given subject")
    response = recommendations.to_dict(orient='records')
    return {"recommendations": response}

# Function to fetch nearby teachers with an extended search if necessary
def get_nearby_teachers(user_lat, user_long, radius_meters, extended_radius=1000):
    # Primary search within the specified radius
    logging.info(f"Searching for teachers within {radius_meters} meters.")
    nearby_teachers = teacher_collection.find({
        "location": {
            "$near": {
                "$geometry": {"type": "Point", "coordinates": [user_long, user_lat]},
                "$maxDistance": radius_meters
            }
        }
    })

    teacher_list = []
    for teacher in nearby_teachers:
        teacher_data = serialize_doc(teacher)  # Convert non-serializable fields
        user_info = get_user_info(teacher["_id"])  # Get user info by teacher's ID
        teacher_data.update(user_info)  # Add user info (name and email) to teacher data
        teacher_list.append(teacher_data)

    # If no teachers found in the primary search, expand the search radius
    if not teacher_list:
        logging.info(f"No teachers found within {radius_meters} meters. Expanding search by an additional 1 km.")
        
        nearby_teachers_extended = teacher_collection.find({
            "location": {
                "$near": {
                    "$geometry": {"type": "Point", "coordinates": [user_long, user_lat]},
                    "$maxDistance": radius_meters + extended_radius
                }
            }
        })
        
        for teacher in nearby_teachers_extended:
            teacher_data = serialize_doc(teacher)  # Convert non-serializable fields
            user_info = get_user_info(teacher["_id"])  # Get user info by teacher's ID
            teacher_data.update(user_info)  # Add user info (name and email) to teacher data
            teacher_list.append(teacher_data)
        
        if teacher_list:
            logging.info(f"Found teachers within {radius_meters + extended_radius} meters.")
        else:
            logging.info(f"No teachers found within {radius_meters + extended_radius} meters either.")

    return teacher_list

# API endpoint to get teachers by location, with extended radius if no results are found
@app.get('/teachers/nearby')
async def teachers_nearby(latitude: float, longitude: float, radius: int = 5000):
    teachers = get_nearby_teachers(latitude, longitude, radius)
    if not teachers:
        raise HTTPException(status_code=404, detail="No teachers found nearby.")
    return JSONResponse(content={"nearby_teachers": teachers})

# Root path
@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}
