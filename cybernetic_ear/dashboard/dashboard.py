from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'cybernetic-ear-secret'
CORS(app)

socketio = SocketIO(
    app,
    async_mode='threading',
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25
)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected to dashboard')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected from dashboard')

def run_dashboard():
    """
    Runs the Flask-SocketIO dashboard.
    """
    print("Starting Flask-SocketIO dashboard on http://localhost:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)
