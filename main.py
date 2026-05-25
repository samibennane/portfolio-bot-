"""Point d'entrée — Portfolio Bot."""

from app import app
import config

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.FLASK_PORT, debug=config.FLASK_DEBUG)