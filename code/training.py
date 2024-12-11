from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from model import JointModel
from data_preprocessing import Features, BookDataset
from load_dataset import load_or_create_dataframe


print(torch.cuda.is_available())
print(torch.cuda.current_device())
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def train(model, train_loader, criterion, optimizer, epochs=10):
#     model.to(device)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
#     for epoch in range(epochs):
#         running_loss = 0.0
#         correct_preds = 0
#         total_preds = 0

#         for image, title, description, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
#             image, title, description, labels = image.to(device), title.to(device), description.to(device), labels.to(device)
           
#             optimizer.zero_grad()

#             outputs = model(image=image, title=title, description=description)

#             loss = criterion(outputs, labels.long())
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             correct_preds += (predicted == labels).sum().item()
#             total_preds += labels.size(0)

#         epoch_loss = running_loss / len(train_loader)
#         epoch_accuracy = correct_preds / total_preds * 100
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        
#         torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
#         scheduler.step()


# def evaluate(model, test_loader):
#     model.eval()
#     correct_preds = 0
#     total_preds = 0
#     running_loss = 0.0
#     criterion = torch.nn.CrossEntropyLoss()

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs[:, :-300], inputs[:, -300:])
#             loss = criterion(outputs, labels.long())
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             correct_preds += (predicted == labels).sum().item()
#             total_preds += labels.size(0)

#     eval_loss = running_loss / len(test_loader)
#     eval_accuracy = correct_preds / total_preds * 100
#     print(f"Test Loss: {eval_loss:.4f}, Test Accuracy: {eval_accuracy:.2f}%")


# """load dataset"""

# image_dir = "C:/Users/hanie/OneDrive/Documents/Hanieh/Master/FOUNDATIONS OF DATA SCIENCE/Project/fds_project/data/images"
# train_csv = "C:/Users/hanie/OneDrive/Documents/Hanieh/Master/FOUNDATIONS OF DATA SCIENCE/Project/fds_project/data/full_data/book_descriptions_train_balanced.csv"
# test_csv = "C:/Users/hanie/OneDrive/Documents/Hanieh/Master/FOUNDATIONS OF DATA SCIENCE/Project/fds_project/data/full_data/book_descriptions_test_balanced.csv"
# saved_train_csv = "C:/Users/hanie/OneDrive/Documents/Hanieh/Master/FOUNDATIONS OF DATA SCIENCE/Project/fds_project/data/full_data/train.csv"
# saved_test_csv = "C:/Users/hanie/OneDrive/Documents/Hanieh/Master/FOUNDATIONS OF DATA SCIENCE/Project/fds_project/data/full_data/test.csv"


# train_df = load_or_create_dataframe(image_dir, train_csv, saved_train_csv)
# test_df = load_or_create_dataframe(image_dir, test_csv, saved_test_csv)

# estimators = [("features", Features(train_df=train_df, test_df=test_df))]
# pipe = Pipeline(estimators)
# pipe.fit([train_df, test_df])
# train_processed, test_processed = pipe.transform([train_df, test_df])

# print(train_processed.head())
# print(train_processed.info())
# print(train_processed.shape)

# train_features = train_processed.drop(columns=["category"])
# test_features = test_processed.drop(columns=["category"])
# train_labels = train_processed["category"]
# test_labels = test_processed["category"]

# train_dataset = BookDataset(train_features, train_labels)
# test_dataset = BookDataset(test_features, test_labels)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# embedding_dim = 300 
# num_classes = len(train_labels.unique()) 
# model = JointModel(embedding_dim, num_classes)

# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = torch.nn.CrossEntropyLoss()

# train(model, train_loader, criterion, optimizer, epochs=10)
# evaluate(model, test_loader)
