# Import required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request, Blueprint
import joblib


# Load course data
course_data = pd.read_csv('Coursera.csv')

# Preprocess data
course_data.drop_duplicates(inplace=True)
course_data.dropna(inplace=True)

# Define features
features = ['Course Name', 'University', 'Difficulty Level', 'Course Rating', 'Course URL', 'Course Description', 'Skills']

# Initialize Flask app
course_app = Blueprint('course', __name__)

# Load trained machine learning model or train and save it if not exists
try:
    model = joblib.load('model_2.joblib')
except FileNotFoundError:
    # Vectorize features
    vectorizer = TfidfVectorizer(stop_words='english')
    course_features = vectorizer.fit_transform(course_data[features].apply(lambda x: ' '.join(x), axis=1))

    # Train machine learning model
    X_train, X_test, y_train, y_test = train_test_split(course_features, course_data['Difficulty Level'], test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))

    # Save model
    joblib.dump(model, 'model_2.joblib')

# Define route for home page
@course_app.route('/')
def home():
    return render_template('jobSearch.html')

# Define route for getting user inputs and displaying recommendations
@course_app.route('/course_recommendations', methods=['POST'])
def get_course_recommendations():

    # Get user inputs
    user_job_title = request.form['job_title']
    user_skills = request.form['skills']

    # Filter course data by user skills
    for skill in user_skills:
        filtered_course_data = course_data[course_data['Skills'].str.contains(skill, case=False)]

    # Vectorize features
    vectorizer = TfidfVectorizer(stop_words='english')
    course_features = vectorizer.fit_transform(filtered_course_data[features].apply(lambda x: ' '.join(x), axis=1))
    user_features = vectorizer.transform([user_job_title + ' ' + user_skills])

    # Train machine learning model
    X_train, X_test, y_train, y_test = train_test_split(course_features, filtered_course_data['Difficulty Level'], test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))

    # Get course recommendations for user
    course_similarity = cosine_similarity(user_features, course_features).flatten()
    filtered_course_data['similarity'] = course_similarity
    filtered_course_data = filtered_course_data.sort_values('similarity', ascending=False)
    c_recommendations = filtered_course_data.head(10)[['Course Name', 'University', 'Difficulty Level', 'Course Rating', 'Course URL', 'Course Description', 'Skills']].values.tolist()
    
    return render_template('jobSearch.html', c_recommendations=c_recommendations)

if __name__ == '__main__':
    course_app.run(debug=True)
