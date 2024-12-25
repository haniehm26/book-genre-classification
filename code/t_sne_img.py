import tensorflow_hub as hub
import numpy as np
import os
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from constants import PATH


df = pd.read_csv(f"{PATH}/data/full_data/book_descriptions_train_balanced.csv")
image_dir = f'{PATH}/data/images'

embed = hub.load("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5")

def preprocess_image(file_path, target_size=(224, 224)):
    """Load and preprocess an image."""
    image = Image.open(file_path).convert('RGB')
    image = image.resize(target_size)
    return np.array(image, dtype=np.float32) / 255.0


df["image_file"] = df["id"].astype(str) + ".jpg" 
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg'))]
df = df[df["image_file"].isin([os.path.basename(f) for f in image_files])]

if len(df) > 10000:
    df = df.sample(n=10000, random_state=42)

image_files = [os.path.join(image_dir, f) for f in df["image_file"]]
images = np.array([preprocess_image(f) for f in image_files], dtype=np.float32)

batch_size = 100
embeddings = []
for i in range(0, len(images), batch_size):
    batch = images[i:i + batch_size]
    batch_embeddings = embed(batch)
    embeddings.append(batch_embeddings.numpy())
embeddings = np.concatenate(embeddings, axis=0)

tsne = TSNE(n_components=2, random_state=42, perplexity=10, n_iter=500)
tsne_embeddings = tsne.fit_transform(embeddings)

categories = df["category"].tolist()
category_list = list(set(categories))
color_map = {category: i for i, category in enumerate(category_list)}
colors = [color_map[cat] for cat in categories]

plt.figure(figsize=(14, 11))
scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=colors, cmap='tab20')
plt.colorbar(scatter, label='Category')

legend_labels = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(i / len(category_list)), 
           markersize=10, label=category) 
    for i, category in enumerate(category_list)
]
plt.legend(handles=legend_labels, title="Categories", bbox_to_anchor=(1.17, 1), loc='upper left', borderaxespad=0.)

plt.title('t-SNE visualization of Image embeddings (colored by category)')
plt.tight_layout()
plt.savefig("t-SNE_visualization_of_image_embeddings.png", dpi=300, bbox_inches="tight")
plt.show()
