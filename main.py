import openai
import os
from openai import OpenAI
import pandas as pd
import requests
from pdf2image import convert_from_path
from io import BytesIO
from PIL import Image
from bs4 import BeautifulSoup
import base64

# Initialize OpenAI client with environment variable for the API key
client = openai.Client(api_key=os.environ.get("OPENAI_API_KEY"))

# Function to convert PDF to images
def pdf_to_images(pdf_url):
    response = requests.get(pdf_url)
    pdf_file = open('temp_menu.pdf', 'wb')
    pdf_file.write(response.content)
    pdf_file.close()

    images = convert_from_path('temp_menu.pdf')
    return images

# Function to extract menu from an image using vision capabilities
def extract_menu_from_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    img_type = 'image/jpeg'

    # Prepare your prompt for extraction
    prompt = "Extract the menu items including name, description, and price from this image."

    # Call to the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"}}
                ]
            }
        ],
        max_tokens=300
    )

    if response.choices and len(response.choices) > 0:
        if hasattr(response.choices[0].message, 'content'):
            formatted_menu = response.choices[0].message.content

            # Attempting manual parsing
            print("Response content:", formatted_menu)  # Log the full content for clarity
            items = []  # To store structured menu items

            # Split the response into lines
            lines = formatted_menu.strip().split('\n')
            for line in lines:
                line = line.strip()  # Clean leading/trailing whitespace

                if " - " in line:  # Assuming a typical pattern
                    name_description = line.split(" - ", 1)
                    if len(name_description) == 2:
                        name = name_description[0].strip()
                        description = name_description[1].strip()

                        # Extract price if it appears at the end (in brackets)
                        price = None
                        if '(' in description and ')' in description:
                            price_part = description.split('(')[-1].strip(')')
                            description = description.split('(')[0].strip()  # Remove price part
                            price = price_part.strip()

                        items.append({'name': name, 'description': description, 'price': price})

            # Convert the items list to a DataFrame and return
            return pd.DataFrame(items)

        else:
            print("Error: Message object does not contain expected content.")
            return pd.DataFrame()
    else:
        print("Error: No choices returned from the API response.")
        return pd.DataFrame()

# Function to check allergens with the LLM
def check_allergens_with_llm(item_name, item_description, user_allergies):
    # Create a prompt for the LLM to assess allergens for a specific item
    prompt = (
        f"The following food item is on the menu:\n"
        f"Name: {item_name}\n"
        f"Description: {item_description}\n"
        "Given the following allergies: " + ", ".join(user_allergies) + ", "
        "does this item potentially contain any of these allergens? Please list any applicable."
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    
    # Extract allergen information from the response
    if response.choices and len(response.choices) > 0:
        return response.choices[0].message.content.strip()

    return None

# Updated ingredient warning function using the LLM
def ingredient_warning_with_llm(menu_df, user_allergies):
    warnings = []

    for idx, row in menu_df.iterrows():
        item_name = row['name']
        item_description = row['description']
        
        allergens_found = check_allergens_with_llm(item_name, item_description, user_allergies)
        
        if allergens_found:
            warnings.append(f"Warning: {item_name} may contain: {allergens_found}")
    
    return warnings

# Function to extract menu information from a website
def extract_menu_from_website(website_url):
    response = requests.get(website_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    menu_dataframes = []
    for img_tag in soup.find_all('img'):  # Adapt the extraction logic based on the actual structure of the webpage
        img_url = img_tag['src']
        img_response = requests.get(img_url)
        image = Image.open(BytesIO(img_response.content))
        menu_df = extract_menu_from_image(image)
        menu_dataframes.append(menu_df)

    return pd.concat(menu_dataframes, ignore_index=True)  # Combine DataFrames if multiple images

# Main function to process input
def main(user_input, user_allergies):
    if user_input.lower().endswith('.pdf'):
        images = pdf_to_images(user_input)
        menu_df = pd.concat([extract_menu_from_image(img) for img in images], ignore_index=True)
    else:
        menu_df = extract_menu_from_website(user_input)

    # Call the new LLM-based ingredient warning function
    warnings = ingredient_warning_with_llm(menu_df, user_allergies)

    print("\nIngredients Warnings:")
    for warning in warnings:
        print(warning)

if __name__ == "__main__":
    user_allergies = ['gluten', 'tomatoes']
    user_input = input("Enter the menu URL or PDF link: ")
    main(user_input, user_allergies)
