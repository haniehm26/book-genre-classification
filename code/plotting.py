from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from wordcloud import WordCloud, STOPWORDS
from matplotlib.lines import Line2D
import pandas as pd
import plotly.express as px
import plotly.io as pio
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("C:/Users/hanie/OneDrive/Documents/Hanieh/Master/FOUNDATIONS OF DATA SCIENCE/Project/fds_project/data/full_data/book_descriptions_train_balanced.csv")

# Distribution of Authors by Number of Books
author_counts = df['author'].value_counts().reset_index()
author_counts = author_counts[author_counts['count'] > 15]
author_counts.columns = ['Author', 'Number of Books']
fig = px.bar(
    author_counts,
    x='Author',
    y='Number of Books',
    title='Distribution of Authors by Number of Books',
    text='Number of Books'
)
fig.show()
# pio.write_image(fig, "Distribution of Authors by Number of Books.png") 


# Average Description Length by Author
df['description_length'] = df['description'].apply(len)
author_counts = df['author'].value_counts().reset_index()
author_counts.columns = ['author', 'count']
filtered_authors = author_counts[author_counts['count'] > 10]['author']
author_avg_length = (
    df[df['author'].isin(filtered_authors)]
    .groupby('author')['description_length']
    .mean()
    .reset_index()
)
author_avg_length['description_length'] = author_avg_length['description_length'].astype(int)
fig = px.bar(
    author_avg_length, 
    x='author', 
    y='description_length', 
    title='Average Description Length by Author',
    labels={'Description Length': 'Average Length', 'Author': 'Authors'},
    text='description_length'
)
fig.show()
# pio.write_image(fig, "Average Description Length by Author.png")


# Average Description Length by Category
category_avg_length = df.groupby('category')['description_length'].mean().reset_index()
category_avg_length['description_length'] = category_avg_length['description_length'].astype(int)
fig = px.bar(
    category_avg_length, 
    x='category', 
    y='description_length', 
    title='Average Description Length by Category',
    labels={'Description Length': 'Average Length', 'Category': 'Categories'},
    text='description_length'
)
fig.show()
# pio.write_image(fig, "Average Description Length by Category.png")


# Show similarities between 150 rondom sampled books based on shared words in titles
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
top_titles = df['title'].sample(n=150, random_state=42)
tfidf_matrix = tfidf_vectorizer.fit_transform(top_titles)
max_len = 20
similarity_matrix = cosine_similarity(tfidf_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=top_titles, columns=top_titles)
similarity_df.columns = [title[:max_len] + '...' if len(title) > max_len else title for title in similarity_df.columns]
similarity_df.index = similarity_df.columns
fig = px.imshow(
    similarity_df,
    labels=dict(x="Book Title", y="Book Title", color="Similarity"),
    x=similarity_df.columns,
    y=similarity_df.index,
    title="Book Title Similarity Heatmap"
)
fig.update_layout(
    title_x=0.5,
    xaxis_tickangle=-45,
    template='plotly_white'
)
fig.show()
# pio.write_image(fig, "Book Title Similarity Heatmap.png")


# Show similarities between 150 rondom sampled books based on shared words in descriptions
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
top_descriptions = df['description'].sample(n=150, random_state=42)
tfidf_matrix = tfidf_vectorizer.fit_transform(top_descriptions)
max_len = 20
similarity_matrix = cosine_similarity(tfidf_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=top_descriptions, columns=top_descriptions)
similarity_df.columns = [description[:max_len] + '...' if len(description) > max_len else description for description in similarity_df.columns]
similarity_df.index = similarity_df.columns
fig = px.imshow(
    similarity_df,
    labels=dict(x="Book Description", y="Book Description", color="Similarity"),
    x=similarity_df.columns,
    y=similarity_df.index,
    title="Book Description Similarity Heatmap"
)
fig.update_layout(
    title_x=0.5,
    xaxis_tickangle=-45,
    template='plotly_white'
)
fig.show()
# pio.write_image(fig, "Book Description Similarity Heatmap.png")


# Top 25 Authors by Total Words in Descriptions
df['word_count'] = df['description'].apply(lambda x: len(x.split()))
author_word_counts = df.groupby('author')['word_count'].sum().reset_index()
author_word_counts = author_word_counts.sort_values(by='word_count', ascending=False)
top_n = 25
top_authors = author_word_counts.head(top_n)
fig = px.bar(top_authors, x='author', y='word_count', 
             title=f"Top {top_n} Authors by Total Words in Descriptions",
             labels={'word_count': 'Total Word Count', 'author': 'Author'},
             color='word_count', 
             color_continuous_scale='Viridis')
fig.show()


# Top 25 Categories by Total Words in Descriptions
df['word_count'] = df['description'].apply(lambda x: len(x.split()))
author_word_counts = df.groupby('category')['word_count'].sum().reset_index()
author_word_counts = author_word_counts.sort_values(by='word_count', ascending=False)
top_n = 25
top_categories = author_word_counts.head(top_n)
fig = px.bar(top_categories, x='category', y='word_count', 
             title=f"Top {top_n} Categories by Total Words in Descriptions",
             labels={'word_count': 'Total Word Count', 'category': 'Category'},
             color='word_count', 
             color_continuous_scale='Viridis')
fig.show()
# pio.write_image(fig, f"Top {top_n} Categories by Total Words in Descriptions.png")


# Word Frequency in Book Titles
text = ' '.join(df['title'].str.strip().str.lower())
filtered_words = [word for word in text.split() if len(word) > 5 and word.isalpha()]
filtered_text = ' '.join(filtered_words)
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(filtered_text)
word_freq = wordcloud.words_
word_freq_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
fig = px.bar(word_freq_df, x='Word', y='Frequency', title="Word Frequency in Book Titles")
fig.update_layout(xaxis_tickangle=-45)
fig.show()
# pio.write_image(fig, "Word Frequency in Book Titles.png")


# Word Frequency in Book Descriptions
text = ' '.join(df['description'].str.strip().str.lower())
filtered_words = [word for word in text.split() if len(word) > 5 and word.isalpha()]
filtered_text = ' '.join(filtered_words)
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(filtered_text)
word_freq = wordcloud.words_
word_freq_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
fig = px.bar(word_freq_df, x='Word', y='Frequency', title="Word Frequency in Book Descriptions")
fig.update_layout(xaxis_tickangle=-45)
fig.show()
# pio.write_image(fig, "Word Frequency in Book Descriptions.png")


# Load the Universal Sentence Encoder model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
df['metadata'] = df['title'] + " " + df['description']
df = df.sample(n=10000, random_state=42)
df = df.dropna(subset=['metadata'])
batch_size = 100 
embeddings = []
for i in range(0, len(df), batch_size):
    batch = df['metadata'][i:i+batch_size]
    batch_embeddings = embed(batch.tolist())
    embeddings.append(batch_embeddings)
embeddings = np.concatenate(embeddings, axis=0)
tsne = TSNE(n_components=2, random_state=42, perplexity=10, n_iter=500)
tsne_embeddings = tsne.fit_transform(embeddings)
category_list = df['category'].unique()
color_map = {category: i for i, category in enumerate(category_list)}
colors = df['category'].map(color_map)
plt.figure(figsize=(14, 11))
scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=colors, cmap='tab20')
plt.colorbar(scatter, label='Category')
legend_labels = [Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(i / len(category_list)), markersize=10, label=category) for i, category in enumerate(category_list)]
plt.legend(handles=legend_labels, title="Categories", bbox_to_anchor=(1.17, 1), loc='upper left', borderaxespad=0.)
plt.title('t-SNE visualization of USE embeddings [title, description] (colored by category)')
plt.tight_layout()
plt.savefig("t-SNE visualization of USE embeddings [title, description] (colored by category).png", dpi=300, bbox_inches="tight")
plt.show()