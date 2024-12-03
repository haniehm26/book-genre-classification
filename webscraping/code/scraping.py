from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from fake_useragent import UserAgent

import random

from constants import PROXY_LISTS

def generate_user_agent():
    return UserAgent().random

def driver_setup():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument('--start-maximized')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument(f"user-agent={generate_user_agent()}")
    # options.add_argument(f'--proxy-server={random.choice(PROXY_LISTS)}')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def scrape_book_details(book_id: str):
    # Set up WebDriver
    driver = driver_setup()

    try:
        # Load Amazon homepage
        driver.get("https://www.amazon.com")
        
        # Wait for the search box
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "twotabsearchtextbox"))
        )
        search_box.send_keys(book_id)
        search_box.submit()
        
        # Wait for search results and locate products
        try:
            products = WebDriverWait(driver, 5).until(
                EC.presence_of_all_elements_located((By.XPATH, '//div[contains(@data-component-type, "s-search-result")]'))
            )
            try:
                # Skip sponsored results if present
                products[0].find_element(By.XPATH, './/span[@aria-label="View Sponsored information or leave ad feedback"]')
            except:
                book_link = products[0].find_element(By.XPATH, './/a[@class="a-link-normal s-no-outline"]').get_attribute("href")
        
                # Navigate to the book details page
                driver.get(book_link)

                # Extract book description from the details page
                try:
                    description = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                        (By.ID, "bookDescription_feature_div"))).text.replace("\n", " ").replace("\t", " ").replace("\r", "")
                    return{"description": description, "status_code": 200}
                except:
                    print(f"No description found for book ID: {book_id}")
                    return{"description": "", "status_code": 200}
        except:
            print(f"No valid book found for ID: {book_id}")
            return {"description": "", "status_code": 200}
    except:
        print("Couldn't load home page.")
        return {"description": "", "status_code": 404}
    
    finally:
        # Quit the driver after processing
        driver.quit()