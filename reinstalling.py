import os

# List of libraries to uninstall and reinstall
libraries = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "joblib",
    "spacy",
    "wordcloud"
]

# Uninstall all libraries
for lib in libraries:
    os.system(f"pip uninstall -y {lib}")

# Install libraries again, with specific version for numpy
for lib in libraries:
    if lib == "numpy":
        os.system(f"pip install numpy==1.26.4")
    else:
        os.system(f"pip install {lib}")

# Specific model for spacy
os.system("python -m spacy download en_core_web_lg")
