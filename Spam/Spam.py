import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

#load data
data = pd.read_csv("E:\\spam.csv", encoding='latin-1')
data = data[['v1','v2']]
data.columns = ['label', 'text']
print(data.head(10))

data['label'] = data['label'].map({'ham': 0, 'spam': 1})
def extract_features(text):
    features = {}
    features['text_length'] = len(text)
    features['special_chars'] = len(re.findall(r'[!$%]', text))
    features['upper_words'] = sum(1 for word in text.split() if word.isupper())
    return features

#Apply feature  extraction
features_df = data['text'].apply(extract_features).apply(pd.Series)


#convert text data to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_features = vectorizer.fit_transform(data['text'])

#combine all features
all_features = np.hstack((tfidf_features.toarray(), features_df.values))

#scale the features
scaler = StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)

#handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(all_features_scaled, data['label'])

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
#Train a logistic Regression classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

#Predict if an email is spam or not
def predict_spam(email_text):
    email_features = extract_features(email_text)
    email_tfidf = vectorizer.transform([email_text])
    email_all_features = np.hstack((email_tfidf.toarray(), np.array([list(email_features.values())])))
    email_all_features_scaled = scaler.transform(email_all_features)
    prediction = model.predict(email_all_features_scaled)
    return 'Spam' if prediction == 1 else 'Not Spam'

#Get email text input from the usee
email_text = input("Enter the email text:")
print("The email is :", predict_spam(email_text))
