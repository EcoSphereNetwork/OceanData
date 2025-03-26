import os
import sys
import logging
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("OceanData.Frontend")

# Erstelle Flask-App
app = Flask(__name__, static_folder='dist')
CORS(app)  # Aktiviere CORS für alle Routen

# Hauptroute für die Frontend-Anwendung
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# API-Endpunkte für die Demo
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "OceanData Frontend Server is running"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    logger.info(f"Starting OceanData Frontend Server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)