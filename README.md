# Sentiment Analysis of Product Reviews

This project performs **Sentiment Analysis** on Amazon product reviews to categorize them as **positive**, **neutral**, or **negative**. We use **Natural Language Processing (NLP)** techniques and machine learning models to analyze the sentiment of reviews.

## Tools and Libraries Used
- **Python**
- **Pandas**: For data manipulation and analysis
- **NLTK/Spacy**: For text preprocessing (tokenization, stop-word removal, stemming)
- **Scikit-Learn**: For vectorization (TF-IDF) and building machine learning models
- **Matplotlib**: For visualizing sentiment distribution
- **WordCloud**: For generating word clouds of frequent words in reviews

## Dataset
We used the **Amazon product reviews** dataset available on **Kaggle**. It contains product reviews along with ratings. The star ratings are mapped to sentiment categories:
- 1-2 stars: Negative
- 3 stars: Neutral
- 4-5 stars: Positive

## Key Steps in the Project

1. **Data Preprocessing**:
   - Removed unnecessary columns from the dataset.
   - Cleaned the review text by removing punctuation, numbers, and stopwords.
   - Applied stemming to reduce words to their base forms.

2. **Text Vectorization**:
   - Used **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert the cleaned text into numerical features for model training.

3. **Model Building**:
   - Built a **Logistic Regression** model to classify reviews into positive, neutral, or negative sentiments.
   - Trained and tested the model using a split of the dataset.

4. **Model Evaluation**:
   - Evaluated the model using **accuracy**, **precision**, **recall**, and **F1-score**.
   - Analyzed misclassified reviews to understand where the model struggled.

5. **Sentiment Distribution**:
   - Visualized the distribution of sentiments (positive, neutral, negative) across the dataset using bar charts.

6. **Word Cloud Visualization**:
   - Generated **Word Clouds** to visualize the most frequent words in each sentiment category, providing insights into customer opinions.

