import pandas as pd
import numpy as np
import string
import nltk
import torch
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from torchvision import transforms
from torch import nn
from transformers import BertTokenizer, BertModel


nltk.download('stopwords')
nltk.download('punkt_tab')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


class Features:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        self.train = train_df
        self.test = test_df
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None) -> None:
        self.label_encoder.fit(self.train['category'])

    def transform(self, X) -> pd.DataFrame:
        self.train = (
            self.train.pipe(self.drop_na_title)
            .pipe(self.drop_na_description)
            .pipe(self.drop_author_column)
            .pipe(self.clean_text)
            .pipe(self.encode_labels)
        )

        self.test = (
            self.test.pipe(self.drop_na_title)
            .pipe(self.drop_na_description)
            .pipe(self.drop_author_column)
            .pipe(self.clean_text)
            .pipe(self.encode_labels)
        )

        return self.train, self.test

    def drop_na_title(self, df: pd.DataFrame) -> pd.DataFrame:
        df.dropna(subset=["title"], inplace=True)
        print("... done with drop_na_title() ...")
        return df

    def drop_na_description(self, df: pd.DataFrame) -> pd.DataFrame:
        df.dropna(subset=["description"], inplace=True)
        print("... done with drop_na_description() ...")
        return df
    
    def drop_author_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop("author", inplace=True, axis=1)
        print("... done with drop_author_column() ...")
        return df
    
    def clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        def clean_text_column(text: str) -> str:
            text = text.translate(str.maketrans(string.digits, " " * len(string.digits)))
            text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
            text = " ".join(text.split())
            words = word_tokenize(text.lower()) 
            stop_words = set(stopwords.words("english"))
            cleaned_text = " ".join([word for word in words if word not in stop_words])
            return cleaned_text

        df['title'] = df['title'].apply(clean_text_column)
        df['description'] = df['description'].apply(clean_text_column)
        print("... done with text cleaning ...")
        return df
    
    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df['category'] = self.label_encoder.transform(df['category'])
        print("... done with label encoding for 'category' ...")
        return df
    

def sentence_to_vector(text: str):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    sentence_embedding = last_hidden_state.mean(dim=1).squeeze()
    return sentence_embedding


class BookDataset(Dataset):
    def __init__(self, features_df: pd.DataFrame, labels: pd.DataFrame):
        self.features = features_df
        self.labels = labels
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
        ])
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        image = self.features.iloc[idx]["image"]
        title = self.features.iloc[idx]["title"]
        description = self.features.iloc[idx]["description"]
        label = self.labels.iloc[idx]

        title_vector = sentence_to_vector(title)
        description_vector = sentence_to_vector(description)

        if self.transform:
            image = self.transform(image)

        image_tensor = torch.tensor(image, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, title_vector, description_vector, label_tensor
    