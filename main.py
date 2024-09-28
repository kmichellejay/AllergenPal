import os
import openai  # Ensure OpenAI is imported
import pandas as pd
import requests
from pdf2image import convert_from_path
import pytesseract
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI  # Import for text LLM
from langchain.agents import initialize_agent, Tool
import json

# Initialize OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize OpenAI LLM for text tasks
text_llm = ChatOpenAI(model="gpt-3.5-turbo", verbose=True)  # Or any other suitable model

# Function to scrape the menu from a website
def scrape_menu_from_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        menu_items = []

        # Update the selector to match the actual structure of the menu in the HTML
        for item in soup.select('.menu-item'):  # Adjust with the correct selector
            name_elem = item.find('h4')
            description_elem = item.find('p', class_='description')
            price_elem = item.find('span', class_='price')

            # Check if elements are found and extract text, handling any potential issues
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
        return pd.DataFrame()  # Return empty DataFrame on error

# Function to handle PDF extraction
def extract_text_from_pdf(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise an error for bad responses

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

        # Further process the combined text into a DataFrame
        menu_items = []
        lines = combined_text.split('\n')
        for line in lines:
            if line.strip():  # Ignore empty lines
                parts = line.split(" - ")  # Assume name and description are split by " - "
                if len(parts) >= 2:
                    name = parts[0].strip()
                    description = parts[1].strip()
                    menu_items.append({'name': name, 'description': description, 'price': None})  # Placeholder

        return pd.DataFrame(menu_items)  # Return a DataFrame

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the PDF: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    except Exception as e:
        print(f"Error processing the PDF: {e}")
        return pd.DataFrame()  # Return empty DataFrame if any other error occurs

# Tool for checking allergens
def check_allergens(item_name, item_description, user_allergies):
    prompt = (
        f"The following food item is on the menu:\n"
        f"Name: {item_name}\n"
        f"Description: {item_description}\n"
        "Given the following allergies: " + ", ".join(user_allergies) + ", "
        "does this item potentially contain any of these allergens? Please list any applicable."
    )
    
    response = text_llm.invoke(prompt)  # Use text LLM for allergen checking
    return response.content.strip()

# Define the checking of allergens tool
def check_allergen_tool(inputs: str) -> str:
    # Parse the input string to extract necessary information
    lines = inputs.split('\n')
    item_name = ""
    item_description = ""
    user_allergies = []
    
    for line in lines:
        if line.startswith("Name:"):
            item_name = line.split("Name:")[1].strip()
        elif line.startswith("Description:"):
            item_description = line.split("Description:")[1].strip()
        elif line.startswith("Given the following allergies:"):
            allergies_part = line.split("Given the following allergies:")[1].strip()
            user_allergies = [allergy.strip() for allergy in allergies_part.split(',')]

    return check_allergens(item_name, item_description, user_allergies)

allergen_tool = Tool(
    name="Check Allergens",
    func=check_allergen_tool,
    description="Check if an item contains allergens."
)

# Initialize tools for LangChain
agent = initialize_agent(
    tools=[allergen_tool],
    llm=text_llm,
    agent_type="chat-zero-shot-react-description",
    verbose=True
)

def main(user_input, user_allergies):
    items = []

    if user_input.lower().endswith('.pdf'):
        items_df = extract_text_from_pdf(user_input)
        items.append(items_df)
    else:
        items_df = scrape_menu_from_website(user_input)
        items.append(items_df)

    menu_df = pd.concat(items, ignore_index=True) if items else pd.DataFrame()

    warnings = {}
    for idx, row in menu_df.iterrows():
        input_prompt = (
            f"The following food item is on the menu:\n"
            f"Name: {row['name']}\n"
            f"Description: {row['description']}\n"
            f"Given the following allergies: {', '.join(user_allergies)}, "
            f"does this item potentially contain any of these allergens? Please list any applicable."
        )

        allergens_found = agent.invoke(input_prompt)
        
        if allergens_found:
            warnings[row['name']] = allergens_found.get('output', '')

    for item, allergen_warning in warnings.items():
        print(f"Warning: {item} may contain: {allergen_warning}")

if __name__ == "__main__":
    user_allergies = ['gluten', 'tomatoes']
    user_input = input("Enter the menu URL or PDF link: ")
    main(user_input, user_allergies)