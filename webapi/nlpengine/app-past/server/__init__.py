from flask import Flask
from flask_cors import CORS
from routes import routesBlueprint
from flask import send_from_directory
import os

server = Flask(__name__)
CORS(server)
server.secret_key = "super secret key"
server.config["SESSION_TYPE"] = "filesystem"
UPLOAD_FOLDER = os.getcwd() + "/uploads"
server.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
server.register_blueprint(routesBlueprint)


@server.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(server.config["UPLOAD_FOLDER"], filename)
