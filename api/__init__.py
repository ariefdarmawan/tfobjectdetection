from flask import Flask
from flask_cors import CORS
api = Flask(__name__, instance_relative_config=True)
CORS(api)
api.config["UPLOAD_FOLDER"] = "upload"
api.config["THRESHOLD_CONFIDENCE"] = 0.70
api.config.from_pyfile('settings.cfg')
from api import routes
routes.__init__()
