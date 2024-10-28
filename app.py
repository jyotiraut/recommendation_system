from flask import Flask, jsonify, request
from pymongo import MongoClient
from bson import ObjectId
import datetime

# Flask app setup
app = Flask(__name__)

# MongoDB connection setup
client = MongoClient("mongodb+srv://anup:1234@cluster0.kuk95.mongodb.net/")
db = client["Tutor"]
teacher_collection = db["teachers"]

# Ensure 2dsphere index on the location field
teacher_collection.create_index([("location", "2dsphere")])

# Helper function to convert ObjectId and datetime fields to strings
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

# Function to fetch nearby teachers based on location
def get_nearby_teachers(user_lat, user_long, radius_meters):
    # Geospatial query to find teachers within the specified radius
    nearby_teachers = teacher_collection.find({
        "location": {
            "$near": {
                "$geometry": {"type": "Point", "coordinates": [user_long, user_lat]},
                "$maxDistance": radius_meters
            }
        }
    })

    # Convert each teacher document to a JSON-serializable format
    teacher_list = [serialize_doc(teacher) for teacher in nearby_teachers]
    return teacher_list

# API endpoint to get teachers by location
@app.route('/teachers/nearby', methods=['GET'])
def teachers_nearby():
    # Retrieve query parameters from request
    user_lat = float(request.args.get('latitude'))
    user_long = float(request.args.get('longitude'))
    radius_meters = int(request.args.get('radius', 5000))  # Default radius to 5000 meters (5 km)

    # Get nearby teachers
    teachers = get_nearby_teachers(user_lat, user_long, radius_meters)

    # Return the response in JSON format
    return jsonify({"nearby_teachers": teachers})

if __name__ == '__main__':
    app.run(debug=True)
