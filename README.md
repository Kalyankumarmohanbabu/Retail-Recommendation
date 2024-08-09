# Retail-Recommendation

## Overview
This project implements a recommendation system for  Retail-Recommendations using machine learning techniques. It features a user-friendly Streamlit interface that allows users to explore products, view recommendations, and customize their search experience.

## Features
- Interactive product selection
- Image display for products
- Personalized product recommendations
- Customizable number of recommendations
- Price and rating information display
- Dataset overview statistics

## Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- Pickle (for model serialization)
- Pillow (for image processing)

## Setup Instructions
1. Clone the repository:

  git clone https://github.com/Kalyankumarmohanbabu/Retail-Recommendation.git
 cd Retail-Recommendation


2. Create and activate a virtual environment:

   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`


3. Install the required packages:
   
   pip install -r requirements.txt


4. Ensure you have the following files in the project directory:
   - `knn_model.pkl`: Pickled KNN pipeline
   - `Amazon-Products.csv`: Dataset containing product information

## Usage
To run the Streamlit app:
```
streamlit run app.py
```

Navigate to the URL provided in the terminal (usually `http://localhost:8501`).

## How It Works
1. Select a product from the sidebar dropdown menu.
2. View the selected product's details and image.
3. Explore recommended products based on the selection.
4. Use the slider to adjust the number of recommendations.
5. Click "Get Custom Recommendations" to update the recommendations.

## Data Preprocessing
The app preprocesses the data to handle currency symbols and commas in price data, ensuring smooth operation of the recommendation system.

## Model
The project uses a K-Nearest Neighbors (KNN) algorithm for generating recommendations based on product features like price and ratings.

## Contributing
Contributions to improve the recommendation system or extend its features are welcome. Please feel free to submit pull requests or open issues for any bugs or enhancements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Data source: [Specify the source of your Amazon product data]
- Inspiration: [Any projects or papers that inspired this work]
