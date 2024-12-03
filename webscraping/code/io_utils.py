import csv

# Function to read book IDs from a CSV file
def read_book_ids(csv_file):
    with open(csv_file, mode='r', encoding='latin-1') as file:
        reader = csv.reader(file)
        # Skip the header and extract the second column
        return [row[1].split(".")[0] for row in reader]
    
# Function to save scraped book description to CSV
def save_to_csv(file_path, book_id, book_data, header):
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["id", "description", "status_code"])
        if header:
            writer.writeheader()
        writer.writerow({"id": book_id, **book_data})