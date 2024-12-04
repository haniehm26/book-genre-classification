from concurrent.futures import ThreadPoolExecutor, as_completed

from scraping import driver_setup, scrape_book_details
from io_utils import read_book_ids, save_to_csv

from constants import BOOK_IDS_FILE, BOOK_DESCRIPTION_FILE, START, END, MAX_WORKERS

import time

def main():
    start_time = time.time()

    driver = driver_setup()

    # Read book IDs
    book_ids = read_book_ids(BOOK_IDS_FILE)[START: END]

    # Initialize book data dictionary
    default_value = {"description": "", "status_code": 404}
    book_data = {book_id: default_value.copy() for book_id in book_ids}

    # Use ThreadPoolExecutor for parallel execution
    header = True
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_book_id = {executor.submit(scrape_book_details, driver, book_id): book_id for book_id in book_ids}
        row = 0
        for future in as_completed(future_to_book_id):
            book_id = future_to_book_id[future]
            try:
                book_detail = future.result()
                book_data[book_id] = book_detail if book_detail is not None else default_value
                print(f"{row} done with book {book_id}")
            except Exception as e:
                print(f"Error scraping book {book_id}: {e}")
                book_data[book_id] = default_value
            # Save scraped data to a CSV file
            save_to_csv(BOOK_DESCRIPTION_FILE, book_id, book_data[book_id], header)
            header = False
            row += 1
  
    driver.quit()
    print(book_data)
    print("Scraping completed. Data saved to file.")

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()