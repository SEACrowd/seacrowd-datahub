rm -rf dist/*
python3 -m build
python3 -m twine upload --repository pypi dist/* --password $PYPI_API_TOKEN --verbose