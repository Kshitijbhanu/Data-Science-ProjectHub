import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv("C:/news.csv")

#Exploratory Data analysis
# Display the first few rows of the dataset
print(df.head())

# Check the dimensions of the dataset (number of rows and columns)
print(df.shape)

# Get information about the dataset including data types and missing values
print(df.info())

# Summary statistics of numerical columns
print(df.describe())

# Exploring Label distribution
# Count the number of instances for each label
print(df['label'].value_counts())

# Visualize the label distribution using plots
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
df['label'].value_counts().plot(kind='bar')
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

#Text Length Analysis
# Calculate text lengths and add as a new column
df['text_length'] = df['text'].apply(lambda x: len(x))

# Summary statistics of text lengths
print(df['text_length'].describe())

# Visualize the distribution of text lengths
plt.figure(figsize=(8, 6))
plt.hist(df['text_length'], bins=30)
plt.title('Text Length Distribution')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

#Word Frequency Analysis
# Tokenize text into words
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Combine text from all instances
all_text = ' '.join(df['text'])

# Tokenize the combined text
tokens = word_tokenize(all_text)

# Count the frequency of each word
word_freq = Counter(tokens)

# Print the most common words
print(word_freq.most_common(20))

#Visualization:
# Visualize the most common words
plt.figure(figsize=(10, 8))
common_words = word_freq.most_common(20)
words, freq = zip(*common_words)
plt.bar(words, freq)
plt.title('Top 20 Most Common Words')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

#N-gram Analysis
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# nltk.download('punkt') # Uncomment if you haven't downloaded the 'punkt' tokenizer

# Function to get top n-grams
def get_top_ngrams(corpus, n=None, ngrams=(1,1)):
    vec = CountVectorizer(ngram_range=ngrams, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Example usage for bi-grams
top_bi_grams = get_top_ngrams(df['text'], n=10, ngrams=(2,2))
print(top_bi_grams)

#Feature Correlation
from sklearn.feature_extraction.text import TfidfVectorizer

# Transforming text to features
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
features = tfidf.fit_transform(df['text']).toarray()

# Creating a dataframe for features (for demonstration, you might need a more efficient approach for large datasets)
features_df = pd.DataFrame(features, columns=tfidf.get_feature_names_out())

# Assuming 'label' is coded as 0 for 'real' and 1 for 'fake'
df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})

# Concatenating features with the label for correlation analysis
full_df = pd.concat([features_df, df['label']], axis=1)

# Calculating correlation
correlation_matrix = full_df.corr()

# Investigating correlation with the target
target_correlation = correlation_matrix['label'].sort_values(ascending=False)
print(target_correlation)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Function to generate and display word cloud
def generate_wordcloud(text_data, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Example text data for real and fake news articles
real_news_text = ' '.join(df[df['label'] == 0]['text'])  # Concatenate text of real news articles
fake_news_text = ' '.join(df[df['label'] == 1]['text'])  # Concatenate text of fake news articles

# Generate word clouds for real and fake news
generate_wordcloud(real_news_text, 'Word Cloud for Real News')
generate_wordcloud(fake_news_text, 'Word Cloud for Fake News')



