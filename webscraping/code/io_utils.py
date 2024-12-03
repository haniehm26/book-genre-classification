import csv

# Function to read book IDs from a CSV file
def read_book_ids(csv_file):
    with open(csv_file, mode='r', encoding='latin-1') as file:
        reader = csv.reader(file)
        # Skip the header and extract the second column
        return [row[1].split(".")[0] for row in reader]
    
# Function to save scraped book description to CSV
def save_to_csv(file_path, book_data):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["id", "description", "status_code"])
        writer.writeheader()
        for book_id, details in book_data.items():
            writer.writerow({"id": book_id, **details})