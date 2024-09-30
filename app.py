from flask import Flask, render_template, request
from main import extract_text_from_pdf, scrape_menu_from_website, process_menu
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        menu_url = request.form['menu_url']
        allergies = request.form['allergies'].split(',')
        allergies = [allergen.strip().lower() for allergen in allergies]

        logging.info(f"Processing menu from: {menu_url}")
        logging.info(f"Checking for allergies: {allergies}")

        try:
            if menu_url.lower().endswith('.pdf'):
                menu_df = extract_text_from_pdf(menu_url)
            else:
                menu_df = scrape_menu_from_website(menu_url)

            if menu_df.empty:
                logging.error("Failed to extract menu information")
                return render_template('error.html', message="Failed to extract menu information. Please check the URL and try again.")

            warnings = process_menu(menu_df, allergies)
            return render_template('results.html', warnings=warnings)
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}", exc_info=True)
            return render_template('error.html', message=f"An error occurred: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)