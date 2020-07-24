import os
from flask import Flask
from flask_assets import Bundle, Environment

bundles = {'javascript':Bundle('simResult.js', output='gen/main.js'),}

def create_app():
    app = Flask(__name__)

    assets = Environment(app)
    assets.register(bundles)

    from . import routes

    app.register_blueprint(routes.bp)

    return app
