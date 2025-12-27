import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Don't pre-load - it uses too much memory
# Just import the app
from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)