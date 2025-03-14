# pytest.ini
[pytest]
markers =
    unit: Markiert Unit-Tests
    integration: Markiert Integrationstests
    performance: Markiert Performance-Tests
    frontend: Markiert Tests für Frontend-Komponenten

# Konfiguration für React-Tests
testpaths = tests
python_files = test_*.py
react_app_root = frontend/

# Umgebungsvariablen für die Tests
env =
    NODE_ENV=test
    REACT_APP_API_URL=http://localhost:5000/api
    REACT_APP_MOCK_API=true

# Plugin-Konfiguration
addopts = --verbose --capture=no
