from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app, async_mode='threading')

@app.route('/')
def index():
    return render_template('index.html')

def run_dashboard():
    """
    Runs the Flask-SocketIO dashboard.
    """
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
