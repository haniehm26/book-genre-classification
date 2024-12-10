from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS
from matplotlib.lines import Line2D
from PIL import Image
import pandas as pd
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os


df = pd.read_csv("C:/Users/hanie/OneDrive/Documents/Hanieh/Master/FOUNDATIONS OF DATA SCIENCE/Project/fds_project/data/full_data/book_descriptions_train_balanced.csv")
image_dir = 'C:/Users/hanie/OneDrive/Documents/Hanieh/Master/FOUNDATIONS OF DATA SCIENCE/Project/fds_project/data/images'


# Word Frequency in Book Titles
text = ' '.join(df['title'].str.strip().str.lower())
filtered_words = [word for word in text.split() if len(word) > 5 and word.isalpha()]
filtered_text = ' '.join(filtered_words)
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', colormap='rainbow').generate(filtered_text)
plt.figure(figsize=(8, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Frequency in Book Titles')
plt.savefig("Word Frequency in Book Titles.png", dpi=300)
plt.show()


# Word Frequency in Book Descriptions
text = ' '.join(df['description'].str.strip().str.lower())
filtered_words = [word for word in text.split() if len(word) > 5 and word.isalpha()]
filtered_text = ' '.join(filtered_words)
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', colormap='rainbow_r').generate(filtered_text)
plt.figure(figsize=(9, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Frequency in Book Descriptions')
plt.savefig("Word Frequency in Book Descriptions.png", dpi=300)
plt.show()


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


# Function to display a grid of book covers
def plot_covers(data, image_dir, rows=3, cols=3, max_title_length=60):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(data):
            img_path = os.path.join(image_dir, f"{data['id'].iloc[i]}.jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax.imshow(img)
                title = data['title'].iloc[i]
                if len(title) > max_title_length:
                    title = title[:max_title_length] + '...'
                ax.set_title(title, fontsize=10 if len(title) <= 20 else 8)
            else:
                ax.text(0.5, 0.5, 'No Image', horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig("Book Covers.png", dpi=300)
    plt.show()
plot_covers(df, image_dir, rows=4, cols=4)

# Function to extract dominant colors from an image
def extract_dominant_colors(data, image_dir, k=7):
    img_path = os.path.join(image_dir, f"{data['id']}.jpg")
    img = Image.open(img_path)
    img = img.resize((img.width // 5, img.height // 5)) 
    img = img.convert('RGB') 
    img_array = np.array(img)
    img_array = img_array.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(img_array)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    total_pixels = img_array.shape[0]
    weights = np.round(counts / total_pixels, 2)
    sorted_indices = np.argsort(-weights)  
    sorted_colors = dominant_colors[sorted_indices] 
    sorted_weights = weights[sorted_indices]     
    return sorted_colors, sorted_weights

# Function to display a grid of book covers with dominant colors and titles
def plot_covers_with_dominant_colors(data, image_dir, rows=4, cols=4, max_title_length=60):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i < len(data):
            ax.axis('off')
            img_path = os.path.join(image_dir, f"{data['id'].iloc[i]}.jpg")
            img = Image.open(img_path)
            sub_fig = ax.inset_axes([0, 0.3, 1, 0.7]) 
            sub_fig.imshow(img)
            sub_fig.axis('off')
            dominant_colors = extract_dominant_colors(data.iloc[i], image_dir)
            sub_colors = ax.inset_axes([0, 0, 1, 0.3]) 
            sub_colors.imshow([dominant_colors], aspect='auto')
            sub_colors.axis('off')
            title = data['title'].iloc[i]
            if len(title) > max_title_length:
                title = title[:max_title_length] + '...'
            ax.set_title(title, fontsize=10 if len(title) <= 20 else 8)
    plt.tight_layout()
    plt.savefig("Book Covers with Dominant Colors.png", dpi=300)
    plt.show()
sampled_df = df.groupby('category').sample(n=1, random_state=1).reset_index()
sampled_df = sampled_df[["category", "id", "title"]].tail(10)
print(sampled_df)
plot_covers_with_dominant_colors(sampled_df, image_dir)


# Compute average dominant colors for each category
def plot_category_color_bars(data, image_dir, k=1):
    category_colors = {}
    for category in data['category'].unique():
        books_in_category = data[data['category'] == category]
        weight_sum = 0
        color_avg = 0
        for _, book in books_in_category.iterrows():
            dominant_colors, weights = extract_dominant_colors(book, image_dir, k=5)
            weight_sum += weights[0]
            color_avg += dominant_colors[0] * weights[0]
        category_colors[category] = color_avg / weight_sum
    num_categories = len(category_colors)
    fig, axes = plt.subplots(1, num_categories, figsize=(num_categories * 3, 3))
    axes = axes.flatten() 
    for i, (category, colors) in enumerate(category_colors.items()):
        dominant_color_strip = np.reshape(colors.astype(int), (1, k, 3))
        wrapped_category = "\n".join(category.split())
        axes[i].imshow(dominant_color_strip, aspect='auto')
        axes[i].set_title(wrapped_category, fontsize=6, rotation=45) 
        axes[i].axis('off')
    plt.subplots_adjust(left=2.9, right=3, top=3, bottom=2.9, wspace=2)
    plt.tight_layout()
    plt.savefig("Dominant Colors of 1000 Books in Each Category.png", dpi=300)
    plt.show()
sampled_df = df.groupby('category').sample(n=1000, random_state=1).reset_index()
sampled_df = sampled_df[["category", "id"]]
plot_category_color_bars(sampled_df, image_dir)
