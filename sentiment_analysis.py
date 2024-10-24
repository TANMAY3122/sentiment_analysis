# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download stopwords
nltk.download('stopwords')

# Load the dataset (replace the path with your dataset location)
df = pd.read_csv('path_to_your_amazon_reviews.csv')

# Step 1: Data Preprocessing
# Remove unwanted columns
df = df[['review_body', 'star_rating']]

# Map star ratings to sentiments (0: negative, 1: neutral, 2: positive)
df['sentiment'] = df['star_rating'].apply(lambda x: 0 if x < 3 else (1 if x == 3 else 2))

# Check for missing values
df = df.dropna()

# Step 2: Text Preprocessing Function
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]  # Remove stop words
    stemmer = SnowballStemmer('english')
    return ' '.join([stemmer.stem(word) for word in words])  # Stemming

# Apply text preprocessing to the review body
df['processed_reviews'] = df['review_body'].apply(preprocess_text)

# Step 3: Vectorization using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['processed_reviews'])

# Step 4: Splitting the Data
y = df['sentiment']  # Sentiment column is the target (positive, neutral, negative)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Building and Training the Model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Misclassifications
misclassified = df.loc[y_test != y_pred, ['review_body', 'sentiment']]
misclassified['predicted_sentiment'] = y_pred[y_test != y_pred]
print("Misclassified Reviews:\n", misclassified.head())

# Step 6: Sentiment Distribution
df['sentiment'].value_counts().plot(kind='bar', color=['green', 'gray', 'red'], title='Sentiment Distribution')
plt.show()

# Step 6: Word Cloud Visualization
positive_reviews = ' '.join(df[df['sentiment'] == 2]['processed_reviews'])
neutral_reviews = ' '.join(df[df['sentiment'] == 1]['processed_reviews'])
negative_reviews = ' '.join(df[df['sentiment'] == 0]['processed_reviews'])

# Positive Word Cloud
positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
plt.figure(figsize=(10,5))
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Positive Reviews')
plt.show()

# Neutral Word Cloud
neutral_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(neutral_reviews)
plt.figure(figsize=(10,5))
plt.imshow(neutral_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Neutral Reviews')
plt.show()

# Negative Word Cloud
negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews)
plt.figure(figsize=(10,5))
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Negative Reviews')
plt.show()
