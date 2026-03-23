# Streamlit Community Cloud automatic entrypoint
import sys
import os

# Execute the dashboard app
dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard', 'app.py')
with open(dashboard_path, 'r', encoding='utf-8') as f:
    code = compile(f.read(), dashboard_path, 'exec')
    exec(code, globals())
