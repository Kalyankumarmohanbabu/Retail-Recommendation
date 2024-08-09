import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image
import requests
from io import BytesIO

# Define paths for model and dataset files
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'D:\\Retail Recommendation\\knn_model.pkl')
dataset_path = os.path.join(base_dir, 'D:\\Retail Recommendation\\Amazon-Products.csv')

# Load the trained model
with open(model_path, 'rb') as f:
    knn_pipeline = pickle.load(f)

# Load and preprocess the dataset
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(dataset_path)
    
    # Clean price columns
    df['discount_price'] = df['discount_price'].replace('₹', '', regex=True).replace(',', '', regex=True).astype(float)
    df['actual_price'] = df['actual_price'].replace('₹', '', regex=True).replace(',', '', regex=True).astype(float)
    
    # Ensure ratings are numeric
    df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')
    
    # Ensure number of ratings are numeric
    df['no_of_ratings'] = pd.to_numeric(df['no_of_ratings'], errors='coerce')
    
    # Handle missing values by filling them with the mean of the respective columns
    df['discount_price'] = df['discount_price'].fillna(df['discount_price'].mean())
    df['actual_price'] = df['actual_price'].fillna(df['actual_price'].mean())
    df['ratings'] = df['ratings'].fillna(df['ratings'].mean())
    df['no_of_ratings'] = df['no_of_ratings'].fillna(df['no_of_ratings'].mean())
    
    # Create a feature matrix for KNN model
    feature_columns = ['discount_price', 'actual_price', 'ratings']
    features = df[feature_columns]
    
    return df, features

# Load dataset and features
df, features = load_and_preprocess_data()

st.title('Retail Recommendation System')

# Search bar for product selection
product_search = st.text_input('Search for a Product', '')

if product_search:
    search_results = df[df['name'].str.contains(product_search, case=False)]
    if not search_results.empty:
        selected_product = st.selectbox('Select a Product', search_results['name'].unique())
        selected_product_details = search_results[search_results['name'] == selected_product].iloc[0]

        st.header(f'Selected Product: {selected_product_details["name"]}')
        st.write(f"Discount Price: ₹{selected_product_details['discount_price']:.2f}")
        st.write(f"Actual Price: ₹{selected_product_details['actual_price']:.2f}")
        st.write(f"Ratings: {selected_product_details['ratings']}")

        # Display the image of the selected product
        if 'image' in selected_product_details and selected_product_details['image']:
            image_url = selected_product_details['image']
            try:
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                st.image(img, caption=selected_product_details['name'], use_column_width=True)
            except:
                st.write("Image not available")
    else:
        st.write("No products found with that name.")

# User input for custom recommendation
st.header('Or Enter Custom Product Details')
discount_price = st.number_input('Discount Price', min_value=0.0, value=10.0)
actual_price = st.number_input('Actual Price', min_value=0.0, value=20.0)
ratings = st.number_input('Ratings', min_value=0.0, max_value=5.0, value=4.0)

if st.button('Get Recommendations'):
    user_features = np.array([[discount_price, actual_price, ratings]])
    
    # Find the nearest neighbors using the loaded KNN model
    distances, indices = knn_pipeline['knn'].kneighbors(knn_pipeline['scaler'].transform(knn_pipeline['imputer'].transform(user_features)))
    
    # Retrieve the recommended products
    recommended_products = df.iloc[indices[0]]
    
    st.header('Recommended Products:')
    for i, row in recommended_products.iterrows():
        st.subheader(f"{i + 1}. {row['name']}")
        st.write(f"Discount Price: ₹{row['discount_price']:.2f}")
        st.write(f"Actual Price: ₹{row['actual_price']:.2f}")
        st.write(f"Ratings: {row['ratings']}")
        
        # Display the image if available
        if 'image' in row and row['image']:
            try:
                response = requests.get(row['image'])
                img = Image.open(BytesIO(response.content))
                st.image(img, caption=row['name'], use_column_width=True)
            except:
                st.write("Image not available")

# Optional: Add more interactive features
st.sidebar.header('Dataset Info')
st.sidebar.write(f"Total products: {len(df)}")
st.sidebar.write(f"Average rating: {df['ratings'].mean():.2f}")
st.sidebar.write(f"Price range: ₹{df['discount_price'].min():,.2f} - ₹{df['discount_price'].max():,.2f}")
