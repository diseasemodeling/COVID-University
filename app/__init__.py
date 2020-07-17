import os
from flask import Flask
from flask_assets import Bundle, Environment



bundles = {
        'd3_viz':Bundle('map.js', 'foresight.js', 'covid-line.js',
                        'userInput-line.js', 'inputlineABC.js', 'policy_builder.js',
                        'simResult.js',
                        output='gen/main.js'),

        'home_css':Bundle('home.css',
                          output='gen/main.css'),
}

ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}

def create_app():
    app = Flask(__name__)


    assets = Environment(app)
    assets.register(bundles)

    from . import models
    from . import routes

    app.register_blueprint(routes.bp)

    return app
