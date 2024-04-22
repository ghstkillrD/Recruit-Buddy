# Import required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request, Blueprint
import joblib
from course import course_app


# Load job listings data
job_data = pd.read_csv('job_data_naukri.csv')

# Preprocess data
job_data.drop_duplicates(inplace=True)
job_data.dropna(inplace=True)

# Define features
features = ['Company', 'Education', 'Experience', 'Industry', 'Job Description', 'Job Location', 'Job Title', 'Pay Rate', 'Shift Type']
job_data['Experience'] = job_data['Experience'].astype(str)


# Initialize Flask app
app = Flask(__name__)

# register the routes for the course Flask app
app.register_blueprint(course_app)

# Load trained machine learning model or train and save it if not exists
try:
    model = joblib.load('model_1.joblib')
except FileNotFoundError:
    # Vectorize features
    vectorizer = TfidfVectorizer(stop_words='english')
    job_features = vectorizer.fit_transform(job_data[features].apply(lambda x: ' '.join(x), axis=1))

    # Train machine learning model
    X_train, X_test, y_train, y_test = train_test_split(job_features, job_data['Industry'], test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))

    # Save model
    joblib.dump(model, 'model_1.joblib')

# Define route for home page
@app.route('/')
def home():
    return render_template('jobSearch.html')

# Define route for getting user inputs and displaying recommendations
@app.route('/recommendations', methods=['POST'])
def get_recommendations():

    # Get user inputs
    user_job_title = request.form['job_title']
    user_shift_type = request.form['shift_type']
    user_experience = request.form['experience']
    user_skills = request.form['skills']
    user_location = request.form['location']
    user_education = request.form['education']

    # Filter job data by shift type
    filtered_job_data = job_data[job_data['Shift Type'] == user_shift_type]

    # Filter job data by experience
    filtered_job_data = filtered_job_data[filtered_job_data['Experience'] <= user_experience]

    # Filter job data by required skills
    for skill in user_skills:
        filtered_job_data = filtered_job_data[filtered_job_data['Job Description'].str.contains(skill, case=False)]

    # Filter job data by location
    for location in user_location:
        filtered_job_data = filtered_job_data[filtered_job_data['Job Location'].str.contains(location, case=False)]
        
    # Filter job data by education
    for education in user_education:
        filtered_job_data = filtered_job_data[filtered_job_data['Education'].str.contains(education, case=False)]

    # Vectorize features
    vectorizer = TfidfVectorizer(stop_words='english')
    job_features = vectorizer.fit_transform(filtered_job_data[features].apply(lambda x: ' '.join(x), axis=1))
    user_features = vectorizer.transform([user_job_title + ' ' + user_skills + ' ' + user_location + ' ' + user_education])

    # Train machine learning model
    X_train, X_test, y_train, y_test = train_test_split(job_features, filtered_job_data['Industry'], test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))

    # Get job recommendations for user
    user_similarity = cosine_similarity(user_features, job_features).flatten()
    filtered_job_data['similarity'] = user_similarity
    filtered_job_data = filtered_job_data.sort_values('similarity', ascending=False)
    recommendations = filtered_job_data.head(10)[['Job Title', 'Shift Type', 'Experience', 'Job Description', 'Company', 'Job Location', 'Pay Rate']].values.tolist()
    
    return render_template('jobSearch.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
