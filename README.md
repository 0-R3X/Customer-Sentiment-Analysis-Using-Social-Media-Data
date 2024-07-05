# Customer-Sentiment-Analysis-Using-Social-Media-Data

# Project Overview
This project involves downloading YouTube comments using the Google Client API, processing the data, training a sentiment analysis model, and creating a user interface (UI) with Streamlit for real-time sentiment analysis.

# Files Included
youtube_comments.csv: Raw comments downloaded from YouTube.
processed_comments.csv: Processed comments ready for model training.
reinstalling.py: Script to uninstall and reinstall necessary libraries, including downgrading numpy. This step ensures the model loads correctly in the Streamlit UI.
ui-streamlit.py: Streamlit-based UI for interacting with the trained sentiment analysis model.
model.joblib: The trained Naive Bayes sentiment analysis model.
Customer Sentiment Analysis Using Social Media Data.ipynb: Jupyter notebook containing data processing and model building code.

# Steps to Run the UI
Install Streamlit: If you haven't already, install Streamlit using pip.

bash
pip install streamlit
Run Reinstalling Script: Ensure that all necessary libraries are properly installed and numpy is downgraded by running the reinstalling.py script.

bash
python reinstalling.py
Run the Streamlit UI: Start the Streamlit UI by executing the following command in the terminal:

bash
streamlit run ui_streamlit.py
Project Workflow
Downloading Comments: Comments are downloaded from YouTube using the Google Client API and saved into youtube_comments.csv.

Processing Data: Data is processed and cleaned, with the processed data saved into processed_comments.csv. This step includes removing unnecessary characters, tokenizing, and other preprocessing tasks.

Training the Model: The Naive Bayes model is trained using the processed data. The training process and data handling are documented in Customer Sentiment Analysis Using Social Media Data.ipynb.

Saving the Model: The trained model is saved as model.joblib for later use.

Building the UI: The Streamlit UI is built in ui-streamlit.py, allowing users to input comments and get sentiment predictions using the trained model.

Troubleshooting
If the model doesn't load correctly in the Streamlit UI, ensure you have run reinstalling.py to reinstall the necessary libraries and downgrade numpy.
By following the above steps, you should be able to run the sentiment analysis model and interact with it through the Streamlit UI seamlessly.

.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
made by R3X
