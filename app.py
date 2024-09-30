from flask import Flask, render_template, request
from main import extract_text_from_pdf, scrape_menu_from_website, process_menu

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        menu_url = request.form['menu_url']
        allergies = request.form['allergies'].split(',')
        allergies = [allergen.strip().lower() for allergen in allergies]

        if menu_url.lower().endswith('.pdf'):
            menu_df = extract_text_from_pdf(menu_url)
        else:
            menu_df = scrape_menu_from_website(menu_url)

        if menu_df.empty:
            return "Failed to extract menu information."

        warnings = process_menu(menu_df, allergies)
        return render_template('templates/results.html', warnings=warnings)

    return render_template('templates/index.html')

if __name__ == '__main__':
    app.run(debug=True)