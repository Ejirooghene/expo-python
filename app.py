from bson import ObjectId
from flask import Flask, jsonify
from dotenv import load_dotenv
import os
from flask_cors import CORS
from pymongo.mongo_client import MongoClient
from sklearn.neighbors import NearestNeighbors
import pandas as pd
load_dotenv()
# import numpy as np

app = Flask(__name__)
CORS(app)
uri = os.getenv("MONGODB_URI")

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

exposure = client['exposure']
users = exposure['users']
exhibits = exposure['exhibits'] 

#FUNCTION FOR CONTENT FILTERING
def content_filter(user_preferences, exhibits):
    # Convert user preferences to lowercase
    user_preferences = set(category.lower() for category in user_preferences)
    
    # Recommend items based on at least one matching category with user preferences
    recommended_exhibits = []
    for exhibit in exhibits:
        exhibit_categories = set(exhibit['category'].lower().split())
        if exhibit_categories.intersection(user_preferences):
            recommended_exhibits.append(exhibit)
            
    if not recommended_exhibits:
        return exhibits
    
    return recommended_exhibits

# FUNCTION TO CONVERT THE DATA
def convert(users, exhibits):
    df_exhibits = pd.DataFrame(exhibits).rename(columns={'_id': 'exhibitId'})
    df_users = pd.DataFrame(users).rename(columns={'_id': 'id'})
    df_exhibits['exhibitId'] = df_exhibits['exhibitId'].astype(str)
    df_users['id'] = df_users['id'].astype(str)


    interaction_data = []


    for user in users:
        user_id = user['_id']
        cart_set, favorite_set, purchase_set = map(set, (user['cart'], user['favorite'], user['purchase']))

        for exhibit_id in set().union(favorite_set, cart_set, purchase_set):
            interaction = 0

            if exhibit_id in favorite_set and exhibit_id not in cart_set and exhibit_id not in purchase_set:
                interaction = 1
            elif exhibit_id in cart_set:
                interaction = 2
            elif exhibit_id in purchase_set:
                interaction = 3

            interaction_data.append({
                'userId': str(user_id),
                'exhibitId': str(exhibit_id),
                'interaction': interaction
            })

    interaction_df = pd.DataFrame(interaction_data)
     # Create a DataFrame with all possible combinations of userId and exhibitId
    all_combinations = pd.DataFrame([(u, e) for u in df_users['id'].unique() for e in df_exhibits['exhibitId']], columns=['userId', 'exhibitId'])

    # Merge with the interaction data to fill missing values with zeros
    unique_combinations = pd.merge(all_combinations, interaction_df, on=['userId', 'exhibitId'], how='left')

    # Fill NaN values with zeros
    unique_combinations['interaction'].fillna(0, inplace=True)

    # Convert the 'interaction' column to integers
    unique_combinations['interaction'] = unique_combinations['interaction'].astype(int)
    pivot = unique_combinations.pivot_table(index='userId', columns='exhibitId', values='interaction')

    return pivot

def recommend_exhibits_for_user(user_id, pivot_df, num_recommendations=1):
    # Create a NearestNeighbors model with cosine similarity
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(pivot_df.values)

    # Get the index of the user in the pivot table
    user_index = pivot_df.index.get_loc(user_id)

    # Find the K nearest neighbors for the user
    distances, indices = model_knn.kneighbors([pivot_df.iloc[user_index].values], n_neighbors=num_recommendations+1)

    # Get user IDs of the nearest neighbors
    recommended_user_indices = pivot_df.index[indices.flatten()][1:num_recommendations+1]

    # Filter the pivot table to include only the specified users
    similar_users_data = pivot_df.loc[recommended_user_indices]

    # Calculate the mean of exhibits for the similar users 
    exhibits_mean = similar_users_data.mean() 

    # Exclude exhibits already interacted with by the target user
    user_interacted_exhibits = pivot_df.loc[user_id]
    exhibits_mean = exhibits_mean[user_interacted_exhibits == 0]

    # Sort the exhibits by the mean value in descending order
    recommended_exhibits = exhibits_mean.sort_values(ascending=False).index[:4]

    return recommended_exhibits

@app.get("/recommend/<user_id>")
def recommend(user_id):
    users_list = list(users.find())
    raw_exhibits = list(exhibits.find())
    user_preferences = users.find_one({"_id": ObjectId(user_id)})["preferences"]
    exhibits_list = content_filter(user_preferences, raw_exhibits)
    pivot = convert(users_list, exhibits_list)
    recommendation = recommend_exhibits_for_user(user_id=user_id, pivot_df=pivot, num_recommendations=2)
    exhibits_values = [dict(exhibit, _id=str(exhibit["_id"])) for exhibit in exhibits_list if str(exhibit["_id"]) in recommendation.tolist()]
    return jsonify(exhibits_values)    
 

if __name__ == '__main__':
    app.run(port=8000, debug=True)    