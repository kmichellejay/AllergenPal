import os
import openai
import pandas as pd
import requests
from pdf2image import convert_from_path
import pytesseract
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
import re
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

# Initialize OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize OpenAI LLM for text tasks
text_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def extract_text_from_pdf(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        with open('temp_menu.pdf', 'wb') as pdf_file:
            pdf_file.write(response.content)

        # Convert PDF pages to images
        images = convert_from_path('temp_menu.pdf')
        extracted_text = []

        # Process each image with Tesseract
        for image in images:
            text = pytesseract.image_to_string(image)
            extracted_text.append(text)

        # Combine extracted text from all images
        combined_text = "\n".join(extracted_text)

        # Process the combined text into a DataFrame
        menu_items = []
        lines = combined_text.split('\n')
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

        return pd.DataFrame(menu_items)

    except Exception as e:
        print(f"Error processing the PDF: {e}")
        return pd.DataFrame()

def scrape_menu_from_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        menu_items = []

        for item in soup.select('.menu-item'):
            name_elem = item.find('h4')
            description_elem = item.find('p', class_='description')
            price_elem = item.find('span', class_='price')

            name = name_elem.text.strip() if name_elem else "Unknown"
            description = description_elem.text.strip() if description_elem else "No description"
            price = price_elem.text.strip() if price_elem else "Price not listed"

            menu_items.append({
                'name': name,
                'description': description,
                'price': price
            })
        return pd.DataFrame(menu_items)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return pd.DataFrame()

def local_allergen_check(item_name, item_description, user_allergies):
    """Perform a local check for obvious allergens before calling the API."""
    detected_allergens = []
    combined_text = (item_name + " " + item_description).lower()
    
    allergen_keywords = {
        'gluten': ['wheat', 'barley', 'rye', 'oats', 'flour', 'bread', 'pasta', 'cereal'],
        'tomatoes': ['tomato', 'marinara', 'ketchup'],
        'dairy': ['milk', 'cheese', 'cream', 'butter', 'yogurt'],
        'nuts': ['peanut', 'almond', 'walnut', 'cashew', 'pistachio'],
        'soy': ['soy', 'tofu', 'edamame'],
        'fish': ['fish', 'salmon', 'tuna', 'cod', 'halibut'],
        'shellfish': ['shrimp', 'crab', 'lobster', 'clam', 'mussel'],
        'eggs': ['egg', 'omelette', 'frittata'],
    }
    
    for allergen, keywords in allergen_keywords.items():
        if allergen in user_allergies and any(keyword in combined_text for keyword in keywords):
            detected_allergens.append(allergen)
    
    return detected_allergens if detected_allergens else None

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def batch_check_allergens(items, user_allergies):
    prompt = "Check the following food items for these allergens: " + ", ".join(user_allergies) + "\n\n"
    for item in items:
        prompt += f"Food item: {item['name']}\nDescription: {item['description']}\n\n"
    prompt += "For each item, list only the allergens from the provided list that are likely present. If none are likely present, say 'No listed allergens detected'. Be concise and separate each item's result with '---'."

    try:
        response = text_llm.invoke(prompt)
        return response.content.strip().split('---')
    except openai.RateLimitError:
        print("Rate limit reached. Waiting before retrying...")
        time.sleep(20)  # Wait for 20 seconds before retrying
        raise  # Re-raise the exception to trigger a retry

def process_menu(menu_df, user_allergies):
    warnings = {}
    items_to_check = []
    
    for _, row in tqdm(menu_df.iterrows(), total=len(menu_df), desc="Pre-processing menu items"):
        name = re.sub(r'[^a-zA-Z\s]', '', row['name']).strip()
        description = row['description'].strip()
        
        if not name or not description:
            continue
        
        local_allergens = local_allergen_check(name, description, user_allergies)
        if local_allergens:
            warnings[name] = ", ".join(local_allergens)
        else:
            items_to_check.append({'name': name, 'description': description})
    
    # Process items in batches
    batch_size = 5
    for i in tqdm(range(0, len(items_to_check), batch_size), desc="Checking allergens with API"):
        batch = items_to_check[i:i+batch_size]
        try:
            results = batch_check_allergens(batch, user_allergies)
            for item, result in zip(batch, results):
                if result.lower() != 'no listed allergens detected':
                    warnings[item['name']] = result.strip()
            time.sleep(2)  # Add a small delay between batches
        except Exception as e:
            print(f"Error processing batch: {e}")
    
    return warnings

def main():
    user_input = input("Enter the menu URL or PDF link: ")
    user_allergies = input("Enter allergies to check (comma-separated, e.g., gluten,tomatoes,dairy): ").split(',')
    user_allergies = [allergen.strip().lower() for allergen in user_allergies]

    if user_input.lower().endswith('.pdf'):
        menu_df = extract_text_from_pdf(user_input)
    else:
        menu_df = scrape_menu_from_website(user_input)

    if menu_df.empty:
        print("Failed to extract menu information.")
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