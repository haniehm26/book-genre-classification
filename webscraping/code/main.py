from scraping import driver_setup, scrape_book_details
from io_utils import read_book_ids, save_to_csv

from constants import BOOK_IDS_FILE, BOOK_DESCRIPTION_FILE

def main():
    # Read book IDs
    book_ids = read_book_ids(BOOK_IDS_FILE)[: 5]

    # Initialize book data dictionary
    default_value = {"description": "", "status_code": 404}
    book_data = {book_id: default_value.copy() for book_id in book_ids}

    # Set up WebDriver
    driver = driver_setup()

    # Scrape book details for each book ID
    for book_id in book_ids:
        book_detail = scrape_book_details(driver, book_id)
        book_data[book_id] = book_detail if book_detail is not None else default_value
        print(f"done with book {book_id}")
    print(book_data)
    # Quit the driver after processing
    driver.quit()

    # Save scraped data to a CSV file
    save_to_csv(BOOK_DESCRIPTION_FILE, book_data)
    print("Scraping completed. Data saved to file.")

if __name__ == "__main__":
    main()