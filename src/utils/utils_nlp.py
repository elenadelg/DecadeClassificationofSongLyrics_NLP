import re
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
import gensim
import pyLDAvis.gensim as gensimvis
from gensim import models

'''''''''''''''''''''''''''''''''''
' WorldCloud                       ' 
'''''''''''''''''''''''''''''''''''

def generate_wordclouds( 
    df, 
    text_column, 
    group_by_column, 
    tfidf_output=False, 
    wordcloud_palette='viridis'
):
    """
    Performs TF-IDF vectorization, and generates Word Clouds grouped by a specific column.
    Displays two word clouds per row.
    """
    # TF-IDF Vectorization
    print("Performing TF-IDF vectorization...")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])
    
    # Generate Word Clouds for each group
    groups = df[group_by_column].unique()
    print(f"Generating Word Clouds grouped by {group_by_column}...")

    # Plot two word clouds per row
    fig, axes = plt.subplots(len(groups) // 2 + len(groups) % 2, 2, figsize=(16, 8))
    axes = axes.flatten()

    for i, group in enumerate(groups):
        group_text = ' '.join(df[df[group_by_column] == group][text_column])
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=wordcloud_palette).generate(group_text)

        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f'Word Cloud for {group}')
        axes[i].axis('off')

    # Hide empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    if tfidf_output:
        return tfidf_matrix, tfidf_vectorizer



'''''''''''''''''''''''''''''''''''
' Sentiment Analysis              '
'''''''''''''''''''''''''''''''''''

class TextSentimentAnalyzer:

    def __init__(self):
        pass

    def analyze_sentiment(self, text):
        """
        Analyzes sentiment using TextBlob library to calculate sentiment polarity:
        - Negative if polarity < 0
        - Neutral if polarity == 0
        - Positive if polarity > 0
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # Get the sentiment polarity

        if polarity > 0:
            return 'positive'
        elif polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def plot_sentiment(self, df, text_column, group_by_column):
        """
        sentiment analysis on text data grouped by a specified column 
        """
        sentiment_results = []

        # Perform sentiment analysis for each group
        for group in df[group_by_column].unique():
            group_df = df[df[group_by_column] == group].copy()
            group_df['sentiment'] = group_df[text_column].apply(self.analyze_sentiment)
            
            sentiment_counts = group_df['sentiment'].value_counts(normalize=True)
            sentiment_results.append({
                group_by_column: group,
                'positive': sentiment_counts.get('positive', 0),
                'neutral': sentiment_counts.get('neutral', 0),
                'negative': sentiment_counts.get('negative', 0)
            })

        sentiment_df = pd.DataFrame(sentiment_results)
        sentiment_df = sentiment_df.sort_values(by=group_by_column)

        colors = {'positive': 'green', 'neutral': 'yellow', 'negative': 'red'}
        sentiment_df.set_index(group_by_column).plot(
            kind='bar', 
            stacked=True,
            color=[colors['positive'], colors['neutral'], colors['negative']]
        )
        plt.title(f'Sentiment Analysis by {group_by_column.capitalize()}')
        plt.xlabel(group_by_column.capitalize())
        plt.ylabel('Percentage')
        plt.xticks(rotation=45 if group_by_column == 'decade' else 0)
        plt.legend(title='Sentiment', loc='upper right', bbox_to_anchor=(1.2, 1))
        plt.show()

    def plot_sentiment_overdecades(self, df, text_column='lyrics_processed'):
        """
        Perform sentiment analysis on lyrics grouped by 'decade' and 
        plot the percentage of sentiments over the decades as a line chart.
        """
        sentiment_results = []

        # For each decade, analyze sentiment
        for decade in df['decade'].unique():
            decade_df = df[df['decade'] == decade].copy()
            decade_df['sentiment'] = decade_df[text_column].apply(self.analyze_sentiment)
            
            sentiment_counts = decade_df['sentiment'].value_counts(normalize=True) * 100
            sentiment_results.append({
                'decade': decade,
                'positive': sentiment_counts.get('positive', 0),
                'neutral': sentiment_counts.get('neutral', 0),
                'negative': sentiment_counts.get('negative', 0)
            })

        # line chart
        sentiment_df = pd.DataFrame(sentiment_results).sort_values(by='decade')
        colors = {'positive': 'lightgreen', 'neutral': 'yellow', 'negative': 'salmon'}
        plt.plot(sentiment_df['decade'], sentiment_df['positive'], marker='o',
                 color=colors['positive'], label='Positive')
        plt.plot(sentiment_df['decade'], sentiment_df['neutral'], marker='o',
                 color=colors['neutral'], label='Neutral')
        plt.plot(sentiment_df['decade'], sentiment_df['negative'], marker='o',
                 color=colors['negative'], label='Negative')
        plt.title('Percentage of Sentiments Over Decades')
        plt.xlabel('Decade')
        plt.ylabel('Percentage')
        plt.xticks(rotation=0)
        plt.legend()
        plt.grid(True)
        plt.show()

        return sentiment_df



'''''''''''''''''''''''''''''''''''
' Topic Modeling                  '
'''''''''''''''''''''''''''''''''''


def topic_modeling(
    df, 
    text_column, 
    no_below=10, 
    keep_n=100000, 
    num_topics=15, 
    iterations=20
):
    """
    Runs topic modeling on the given DataFrame using Gensim dictionary class
    and returns a PyLDAvis visualization object.

    """
    
    processed_lyrics = df[text_column].str.split()
    dictionary = gensim.corpora.Dictionary(processed_lyrics) # Gensim dictionary
    dictionary.filter_extremes(no_below=no_below, keep_n=keep_n)  # filter extreme terms

    # Convert each document to a Bag-of-Words representation
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_lyrics] #

    # TF-IDF model from the BoW corpus
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    # Train LDA model
    lda_model = gensim.models.LdaMulticore(
        bow_corpus, 
        num_topics=num_topics,
        id2word=dictionary, 
        iterations=iterations
    )

    # prepare for visualization with pyLDAvis
    vis_data = gensimvis.prepare(lda_model, bow_corpus, dictionary)

    return vis_data
