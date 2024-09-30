import os
import openai
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
import re
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
import logging
import io
from PyPDF2 import PdfReader
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    logging.error("OPENAI_API_KEY is not set")

# Initialize OpenAI LLM for text tasks
text_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def extract_text_from_pdf(pdf_url):
    logging.info(f"Attempting to extract text from PDF: {pdf_url}")
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        logging.info("Successfully downloaded PDF")

        with io.BytesIO(response.content) as pdf_file:
            reader = PdfReader(pdf_file)
            extracted_text = ""
            for page in reader.pages:
                extracted_text += page.extract_text() + "\n"

        logging.info(f"Extracted {len(extracted_text)} characters from PDF")

        # Process the extracted text into a DataFrame
        menu_items = []
        lines = extracted_text.split('\n')
        current_item = {}
        for line in lines:
            line = line.strip()
            if line:
                # Check if the line is a new menu item (typically in all caps or starts with a number)
                if line.isupper() or re.match(r'^\d+\.?\s', line):
                    if current_item:
                        menu_items.append(current_item)
                    current_item = {'name': line, 'description': ''}
                elif current_item:
                    # Ignore prices and other non-descriptive lines
                    if not re.match(r'^\$?\d+(\.\d{2})?$', line) and len(line) > 3:
                        current_item['description'] += ' ' + line

        if current_item:
            menu_items.append(current_item)

        logging.info(f"Extracted {len(menu_items)} menu items from PDF")
        return pd.DataFrame(menu_items)

    except Exception as e:
        logging.error(f"Error processing the PDF: {e}", exc_info=True)
        return pd.DataFrame()

def scrape_menu_from_website(url):
    logging.info(f"Attempting to scrape menu from website: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        logging.info("Successfully fetched webpage")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        menu_items = []

        # This is a generic scraper and might need to be adjusted based on the specific website structure
        for item in soup.select('div.menu-item, li.menu-item'):
            name_elem = item.find(['h3', 'h4', 'strong', 'span'], class_=['item-name', 'menu-item-name'])
            description_elem = item.find(['p', 'span'], class_=['description', 'item-description'])
            
            name = name_elem.text.strip() if name_elem else "Unknown"
            description = description_elem.text.strip() if description_elem else "No description"

            menu_items.append({
                'name': name,
                'description': description
            })
        
        if not menu_items:
            logging.warning("No menu items found with specific selectors, attempting to extract all text")
            # If no items found, try to extract all text as a fallback
            all_text = soup.get_text()
            menu_items = [{'name': 'Full Menu', 'description': all_text}]

        logging.info(f"Extracted {len(menu_items)} menu items from website")
        return pd.DataFrame(menu_items)

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching the webpage: {e}", exc_info=True)
        return pd.DataFrame()

@lru_cache(maxsize=1000)
def cached_allergen_check(item_name, item_description, user_allergies_tuple):
    user_allergies = list(user_allergies_tuple)
    return local_allergen_check(item_name, item_description, user_allergies)

def local_allergen_check(item_name, item_description, user_allergies):
    detected_allergens = []
    combined_text = (item_name + " " + item_description).lower()
    
    allergen_keywords = {
        'crab': ['crab', 'kani', 'surimi'],
        'beef': ['beef', 'steak', 'wagyu', 'angus', 'sirloin', 'ribeye', 'tenderloin', 'brisket', 'prime rib'],
        'cheese': ['cheese', 'cheddar', 'mozzarella', 'parmesan', 'gouda', 'brie', 'feta', 'ricotta', 'cream cheese'],
        'gluten': ['wheat', 'barley', 'rye', 'oats', 'flour', 'bread', 'pasta', 'cereal', 'gluten'],
        'dairy': ['milk', 'cream', 'butter', 'yogurt', 'lactose', 'whey', 'casein'],
        'nuts': ['peanut', 'almond', 'walnut', 'cashew', 'pistachio', 'hazelnut', 'macadamia', 'pecan'],
        'soy': ['soy', 'tofu', 'edamame', 'miso', 'tempeh'],
        'fish': ['fish', 'salmon', 'tuna', 'cod', 'halibut', 'tilapia', 'anchovy', 'sardine'],
        'shellfish': ['shrimp', 'lobster', 'clam', 'mussel', 'oyster', 'scallop'],
        'eggs': ['egg', 'omelette', 'frittata', 'mayonnaise', 'meringue'],
    }
    
    for allergen, keywords in allergen_keywords.items():
        if allergen in user_allergies and any(keyword in combined_text for keyword in keywords):
            detected_allergens.append(allergen)
    
    return detected_allergens

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def batch_check_allergens(items, user_allergies):
    prompt = f"""Analyze the following food items for these allergens: {', '.join(user_allergies)}. 
    For each item, list ONLY the allergens from the provided list that are definitely or very likely present. 
    If none are likely present, say 'None detected'. Be thorough and consider indirect ingredients.
    Separate each item's result with '---'.

    Example output format:
    Item name: detected allergens or 'None detected'
    ---
    """
    
    for item in items:
        prompt += f"Food item: {item['name']}\nDescription: {item['description']}\n\n"

    try:
        response = text_llm.invoke(prompt)
        return [result.strip() for result in response.content.strip().split('---') if result.strip()]
    except openai.RateLimitError:
        logging.warning("Rate limit reached. Waiting before retrying...")
        time.sleep(20)
        raise

def process_menu(menu_df, user_allergies):
    warnings = {}
    items_to_check = []
    
    user_allergies_tuple = tuple(user_allergies)
    
    for _, row in tqdm(menu_df.iterrows(), total=len(menu_df), desc="Pre-processing menu items"):
        name = row['name'].strip()
        description = row['description'].strip()
        
        if not name:
            continue
        
        local_allergens = cached_allergen_check(name, description, user_allergies_tuple)
        if local_allergens:
            warnings[name] = ", ".join(local_allergens)
        else:
            items_to_check.append({'name': name, 'description': description})
    
    batch_size = 20  # Adjusted for balance between efficiency and API limits
    for i in tqdm(range(0, len(items_to_check), batch_size), desc="Checking allergens with API"):
        batch = items_to_check[i:i+batch_size]
        try:
            results = batch_check_allergens(batch, user_allergies)
            for item, result in zip(batch, results):
                if ':' in result:
                    item_name, allergens = result.split(':', 1)
                    item_name = item_name.strip()
                    allergens = allergens.strip()
                    if allergens.lower() != 'none detected':
                        warnings[item_name] = allergens
                else:
                    logging.warning(f"Unexpected result format: {result}")
            time.sleep(3)  # Adjusted delay between batches
        except Exception as e:
            logging.error(f"Error processing batch: {e}", exc_info=True)
    
    return warnings

def main():
    user_input = input("Enter the menu URL or PDF link: ")
    user_allergies = input("Enter allergies to check (comma-separated, e.g., gluten,dairy,nuts): ").split(',')
    user_allergies = [allergen.strip().lower() for allergen in user_allergies]

    logging.info(f"Processing menu from: {user_input}")
    logging.info(f"Checking for allergies: {user_allergies}")

    if user_input.lower().endswith('.pdf'):
        menu_df = extract_text_from_pdf(user_input)
    else:
        menu_df = scrape_menu_from_website(user_input)

    if menu_df.empty:
        logging.error("Failed to extract menu information")
        print("Failed to extract menu information. Please check the URL and try again.")
        return

    warnings = process_menu(menu_df, user_allergies)

    if warnings:
        print("\nAllergen Warnings:")
        for item, allergens in warnings.items():
            print(f"- {item}: {allergens}")
    else:
        print("No allergen warnings for the given allergies.")

if __name__ == "__main__":
    main()