import os
import pickle
import pandas as pd
import numpy as np
from PIL import Image


def load_or_create_dataframe(image_dir, metadata_csv, saved_file_path):
    """
    Load a saved DataFrame if it exists, otherwise load from CSV, save it, and return.

    Args:
        metadata_csv (str): Path to the CSV file containing metadata.
        saved_file_path (str): Path to the saved DataFrame file.

    Returns:
        pd.DataFrame: The loaded or newly created DataFrame.
    """
    # Check if a saved DataFrame exists
    df = __load_saved_dataframe(saved_file_path)
    if df is not None:
        # Return the loaded DataFrame if it exists
        return df
    else:
        # Load from the original CSV
        df = __load_images_with_metadata(image_dir, metadata_csv)
        # Save the DataFrame for future use
        __save_dataframe_to_file(df, saved_file_path)
        return df
    

def __load_images_with_metadata(image_dir, metadata_csv, image_size=(224, 224)):
    """
    Load images and metadata, pairing them together.

    Args:
        image_dir (str): Path to the directory containing images.
        metadata_csv (str): Path to the CSV file containing metadata.
        image_size (tuple): Target size for resizing images (width, height).

    Returns:
        list[dict]: A list where each entry contains metadata and the corresponding image array.
    """
    # Load metadata
    print("Loading dataset ...")
    metadata = pd.read_csv(metadata_csv)

    metadata.columns = [
        "id",
        "img_name",
        "img_url",
        "title",
        "author",
        "category_id",
        "category",
        "description",
        "status_code"
    ]

    # List to store paired data
    dataset = []

    for _, row in metadata.iterrows():
        img_name = row['img_name']
        img_path = os.path.join(image_dir, img_name)

        # Check if the image exists
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_name} not found at {img_path}. Skipping.")
            continue

        # Load and preprocess the image
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")  # Ensure 3 channels
                img = img.resize(image_size)  # Resize image
                img_array = np.array(img)  # Convert to NumPy array
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            continue

        # Append data to the dataset
        dataset.append({
            "id": row['id'],
            "image": img_array,
            "title": row['title'],
            "author": row['author'],
            "category": row['category'],
            "description": row['description']
        })
    return pd.DataFrame(dataset)


def __load_saved_dataframe(file_path):
    """
    Load a saved DataFrame from a file.

    Args:
        file_path (str): Path to the saved DataFrame file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    print(f"Checking for existing saved dataset at {file_path}...")
    if os.path.exists(file_path):
        print(f"Saved dataset found. Loading from {file_path}...")
        with open(file_path, 'rb') as file:
            df = pickle.load(file)
        print("Dataset loaded successfully!")
        return df
    else:
        print("No saved dataset found.")
        return None
    
    
def __save_dataframe_to_file(df, file_path):
    """
    Save a DataFrame to a file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): Path to save the DataFrame file.

    Returns:
        None
    """
    print(f"Saving DataFrame to {file_path}...")
    with open(file_path, 'wb') as file:
        pickle.dump(df, file)
    print("DataFrame saved successfully!")