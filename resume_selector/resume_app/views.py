from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
import re
import nltk
import PyPDF2
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the resume dataset
resume_df = pd.read_csv(
    'C:\\Users\\Muhammad Ishfaq\\Desktop\\Python Programming\\Resume_Selection_Project\\004_Resume_Selection_with_ML\\data\\resume_data.csv', encoding='latin-1')

# Preprocess text function


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    filtered_text = [lemmatizer.lemmatize(word.lower(
    )) for word in word_tokens if word.lower() not in stop_words and word.isalnum()]
    return ' '.join(filtered_text)


# Preprocess the resume text in the dataset
resume_df['cleaned_resume_text'] = resume_df['resume_text'].apply(
    preprocess_text)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(resume_df['cleaned_resume_text'])
y = resume_df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train a Multinomial Naive Bayes classifier
Bayes_clf = MultinomialNB(alpha=3)
Bayes_clf.fit(X_train, y_train)

# Create your views here.

# Route for the home page


def home(request):
    return render(request, 'index.html')

# Route for handling resume upload and performing operations


def upload_resume(request):
    if request.method == 'POST' and request.FILES['resume']:
        # Get the uploaded resume file
        resume_file = request.FILES['resume']
        resume_text = resume_file.read().decode('latin-1')

        resume_text = extract_text_from_pdf(resume_file)

        # Analyze the resume and provide suggestions
        feedback = analyze_resume(resume_text)

        # Predict if the resume should be flagged or not
        prediction_info = predict_resume_class_with_info(resume_text)

        return render(request, 'result.html', {'feedback': feedback, 'prediction_info': prediction_info})
    else:
        return render(request, 'index.html')

# Function to analyze resume content and provide suggestions


def analyze_resume(resume_text):
    # Define criteria along with their associated keywords
    criteria = {
        "Summary": ["summary", "objective", "About Me", "profile summary", "personal statement", "executive summary", "career summary"],
        "Education": ["education", "academic background", "qualifications", "educational history", "academic qualifications"],
        "Experience": ["experience", "work", "employment", "work experience", "professional experience", "employment history", "career history", "work history"],
        "Skills": ["skills", "competencies", "language skills", "digital skills", "professional skills", "relevant skills", "skills summary", "technical skills", "core competencies", "key skills", "soft skills"],
        "Projects": ["projects", "academic projects", "professional projects", "work projects", "relevant projects", "project experience"],
        "Certifications": ["certifications", "certificates", "professional certifications", "licenses", "qualifications", "credentials", "certification"],
        "Interest": ["interest", "hobbies", "personal interests", "activities", "extracurricular activities", "volunteer work", "community involvement"],
        # Add more criteria as needed
    }

    # Initialize feedback
    feedback = {}

    # Regular expression to match headings or subheadings
    heading_regex = r'(?i)(?:^|\n)\s*([A-Za-z\s]+)\s*(?:\n|$)'

    # Extract headings from the resume text
    headings = re.findall(heading_regex, resume_text)

    # Analyze each criterion
    for section, keywords in criteria.items():
        found_keywords = []
        for keyword in keywords:
            for heading in headings:
                if keyword.lower() in heading.lower():
                    found_keywords.append(keyword)
                    break  # Exit loop once a matching heading is found
        if found_keywords:
            feedback[section] = "Found: " + ", ".join(found_keywords)
        else:
            feedback[section] = "Missing: Consider adding information about " + \
                section.lower() + "."

    return feedback

# Function to extract text from PDF


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


# Function to predict resume class and provide additional information


def predict_resume_class_with_info(resume_text):
    cleaned_resume = preprocess_text(resume_text)
    transformed_resume = vectorizer.transform([cleaned_resume])
    prediction = Bayes_clf.predict(transformed_resume)
    prediction_probability = Bayes_clf.predict_proba(transformed_resume)

    if prediction[0] == 1:
        prediction_result = "Flagged"
        explanation = "This resume is flagged, which means it may contain elements that require further review."
    else:
        prediction_result = "Not Flagged"
        explanation = "This resume is not flagged, indicating it appears to meet standard criteria."

    return {
        "Prediction": prediction_result,
        "Prediction_Probability": {
            "Not_Flagged": prediction_probability[0][0],
            "Flagged": prediction_probability[0][1]
        },
        "Explanation": explanation
    }
