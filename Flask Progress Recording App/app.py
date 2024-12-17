from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime

app = Flask(__name__)

# In-memory storage for updates and history
updates = []
history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about', methods=['GET', 'POST'])
def about():
    if request.method == 'POST':
        contact_name = request.form.get('name')
        contact_message = request.form.get('message')
        # Process the contact form (e.g., send an email or store in a database)
        return redirect(url_for('about'))
    return render_template('about.html')

@app.route('/progress', methods=['GET', 'POST'])
def progress():
    if request.method == 'POST':
        progress_update = request.form.get('progress')
        # Save the progress update (e.g., store in a database or in-memory list)
        updates.append({'update': progress_update, 'timestamp': datetime.now()})
        return redirect(url_for('progress'))
    return render_template('progress.html', updates=updates)

@app.route('/history', methods=['GET', 'POST'])
def history_page():
    if request.method == 'POST':
        new_entry = request.form.get('entry')
        # Add the new entry to history
        history.append({'entry': new_entry, 'timestamp': datetime.now()})
        return redirect(url_for('history_page'))
    return render_template('history.html', history=history)

@app.route('/surveillance')
def surveillance():
    return render_template('surveillance.html', timestamp=datetime.now())

if __name__ == '__main__':
    app.run(debug=True)
