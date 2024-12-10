import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("C:/Users/hanie/OneDrive/Documents/Hanieh/Master/FOUNDATIONS OF DATA SCIENCE/Project/fds_project/data/full_data/book_descriptions_train_balanced.csv")


# Number of Books by Category Pie Chart
category_counts = df['category'].value_counts().reset_index()
category_counts.columns = ['Category', 'Number of Books']
fig = px.pie(
    category_counts,
    names='Category',
    values='Number of Books',
    title='Distribution of Books by Category',
    color_discrete_sequence=px.colors.qualitative.Set3
)
fig.show()
# pio.write_image(fig, "Number of Books by Category - Pie Chart.png") 


# Number of Books by Category Bar Chart
fig = px.bar(
    category_counts,
    x='Category',
    y='Number of Books',
    title='Number of Books by Category',
    text='Number of Books'
)
fig.show()
# pio.write_image(fig, "Number of Books by Category - Bar Chart.png") 


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