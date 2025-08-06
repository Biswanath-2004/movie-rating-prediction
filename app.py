import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import os

# Set up the Streamlit app
st.title("Movie Rating Prediction")
st.write("This app predicts movie ratings based on various features from the IMDb dataset.")

def load_data():
    """Load and preprocess the movie data"""
    # Check if dataset exists
    if not os.path.exists('IMDb_Movies_India.csv'):
        st.error("Dataset file 'IMDb_Movies_India.csv' not found in the current directory.")
        st.stop()
    
    try:
        # Load the dataset with latin-1 encoding
        df = pd.read_csv('IMDb_Movies_India.csv', encoding='latin-1')
        
        # Basic preprocessing (same as in the notebook)
        df.fillna(method='ffill', inplace=True)
        
        # Encode categorical features
        label_encoders = {}
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        if 'Rating' in object_cols:
            object_cols.remove('Rating')
        
        for col in object_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        
        # Prepare features and target
        X = df.drop(columns=['Rating'])
        y = df['Rating']
        
        # Convert y to numeric, coercing non-numeric values to NaN
        y = pd.to_numeric(y, errors='coerce')
        
        # Drop rows with NaN values in y from both X and y
        X = X[y.notna()]
        y = y[y.notna()]
        
        return X, y, label_encoders, df
    
    except Exception as e:
        st.error(f"Error loading or processing the dataset: {str(e)}")
        st.stop()

def train_model(X, y):
    """Train the Random Forest model"""
    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2, X_train.columns

def show_feature_importance(model, features):
    """Display feature importance plot"""
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances, y=features)
    plt.title("Feature Importance for Movie Rating Prediction")
    st.pyplot(plt)

def predict_rating(model, label_encoders, feature_names):
    """Create a form for users to input movie details and predict rating"""
    st.subheader("Predict Movie Rating")
    
    with st.form("movie_form"):
        st.write("Enter movie details:")
        
        # Create input fields for each feature
        inputs = {}
        for feature in feature_names:
            if feature in label_encoders:
                # For categorical features that were encoded
                options = label_encoders[feature].classes_
                inputs[feature] = st.selectbox(feature, options)
            else:
                # For numerical features
                inputs[feature] = st.number_input(feature)
        
        submitted = st.form_submit_button("Predict Rating")
        
        if submitted:
            # Prepare input data for prediction
            input_data = []
            for feature in feature_names:
                if feature in label_encoders:
                    # Encode categorical features
                    encoded_value = label_encoders[feature].transform([inputs[feature]])[0]
                    input_data.append(encoded_value)
                else:
                    input_data.append(inputs[feature])
            
            # Convert to numpy array and reshape
            input_array = np.array(input_data).reshape(1, -1)
            
            # Make prediction
            predicted_rating = model.predict(input_array)[0]
            
            st.success(f"Predicted Rating: {predicted_rating:.2f}")

def main():
    # Load and preprocess data
    X, y, label_encoders, df = load_data()
    
    # Train model
    model, mse, r2, feature_names = train_model(X, y)
    
    # Display model metrics
    st.subheader("Model Performance")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
    
    # Show feature importance
    show_feature_importance(model, feature_names)
    
    # Show sample of the data
    if st.checkbox("Show sample data"):
        st.write(df.head())
    
    # Prediction form
    predict_rating(model, label_encoders, feature_names)

if __name__ == "__main__":
    main()