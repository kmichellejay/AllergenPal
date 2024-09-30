# AllergenPal
 Allergy friendly menu scanner
 Must have Poppler installed to run locally
 Install Homebrew and then install Poppler and tesseract

 also set your openai api key in your environement first.
 Vision Agent: This will read the menu from either a PDF or a website and extract the relevant information into a structured format (e.g., a DataFrame).
Allergen Filtering Agent: This component will take a list of allergens from the user, cross-reference it with the menu items, and filter out items that the user should avoid.
Knowledge-Based Warning Agent: This will use contextual knowledge of common ingredients in certain dishes to provide further warnings about potential allergens based on typical dish compositions.
