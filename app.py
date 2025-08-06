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

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .title {
            color: #2c3e50;
            text-align: center;
            padding: 15px;
            border-bottom: 2px solid #3498db;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
            border: none;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #2980b9;
            transform: scale(1.02);
        }
        .stSelectbox, .stNumberInput {
            margin-bottom: 15px;
        }
        .success-box {
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            border-left: 5px solid #28a745;
        }
        .metric-box {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .feature-importance-header {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Set up the Streamlit app with improved layout
st.markdown('<h1 class="title">🎬 Movie Rating Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center; color: #7f8c8d; font-size: 16px;'>
        Predict IMDb movie ratings based on various features like genre, director, actors, and more.
    </p>
""", unsafe_allow_html=True)

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
    st.markdown('<h2 class="feature-importance-header">🔍 Feature Importance</h2>', unsafe_allow_html=True)
    importances = model.feature_importances_
    
    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot using Seaborn with custom style
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Importance', y='Feature', data=importance_df, palette="Blues_d")
    plt.title("Feature Importance for Movie Rating Prediction", pad=20, fontsize=14)
    plt.xlabel('Importance Score', labelpad=10)
    plt.ylabel('Features', labelpad=10)
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(plt)

def predict_rating(model, label_encoders, feature_names):
    """Create a form for users to input movie details and predict rating"""
    st.markdown('<h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">🔮 Predict Movie Rating</h2>', unsafe_allow_html=True)
    
    with st.form("movie_form"):
        # Create a two-column layout for better form organization
        col1, col2 = st.columns(2)
        
        inputs = {}
        for i, feature in enumerate(feature_names):
            # Alternate between columns for better layout
            current_col = col1 if i % 2 == 0 else col2
            
            with current_col:
                if feature in label_encoders:
                    # For categorical features that were encoded
                    options = label_encoders[feature].classes_
                    inputs[feature] = st.selectbox(feature, options, key=feature)
                else:
                    # For numerical features
                    inputs[feature] = st.number_input(feature, key=feature)
        
        # Center the submit button
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🎯 Predict Rating")
        st.markdown("</div>", unsafe_allow_html=True)
        
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
            
            # Display prediction with nice styling
            st.markdown(f"""
                <div class="success-box">
                    <h3 style='color: #28a745; margin-top: 0;'>Prediction Result</h3>
                    <p style='font-size: 18px;'>The predicted IMDb rating for this movie is: <strong>{predicted_rating:.2f}/10</strong></p>
                </div>
            """, unsafe_allow_html=True)

def main():
    # Load and preprocess data
    X, y, label_encoders, df = load_data()
    
    # Train model
    model, mse, r2, feature_names = train_model(X, y)
    
    # Display model metrics in a nice layout
    st.markdown('<h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">📊 Model Performance</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="metric-box">
                <h3 style='color: #2c3e50; margin-top: 0;'>Mean Squared Error</h3>
                <p style='font-size: 24px; color: #3498db;'>{mse:.2f}</p>
            </div>
        """.format(mse=mse), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-box">
                <h3 style='color: #2c3e50; margin-top: 0;'>R² Score</h3>
                <p style='font-size: 24px; color: #3498db;'>{r2:.2f}</p>
            </div>
        """.format(r2=r2), unsafe_allow_html=True)
    
    # Show feature importance
    show_feature_importance(model, feature_names)
    
    # Show sample of the data with a toggle
    if st.checkbox("🎥 Show sample data", key="show_data"):
        st.markdown('<h3 style="color: #2c3e50; margin-top: 20px;">📋 Sample Data</h3>', unsafe_allow_html=True)
        st.dataframe(df.head().style.background_gradient(cmap='Blues'))
    
    # Prediction form
    predict_rating(model, label_encoders, feature_names)

if __name__ == "__main__":
    main()
