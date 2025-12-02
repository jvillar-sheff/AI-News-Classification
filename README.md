# 📰 NLP News Classification App

This project demonstrates a news article classifier built using a fine-tuned **DistilBERT** model on the AG News dataset, deployed via a **Streamlit** web application.

## 1. Setup and Installation

### 1.1. Clone the Repository

```bash
git clone https://github.com/jvillar-sheff/AI-News-Classification.git
cd AI-News-Classification
```
### 1.2. Create and Activate Virtual Environment (venv)

For macOS/Linux (Bash/Zsh):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```
For Windows (PowerShell):
```bash
# Assuming 'python' maps to 3.12, otherwise use 'py -3.12'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 1.3. Install Dependencies
Install all necessary libraries using the provided requirements.txt file. Ensure your venv is active before running this command.

```bash
pip install -r requirements.txt
```

## 2. Run the Notebooks (Data Preparation & Training)
The core data processing and model fine-tuning steps must be executed first to generate the necessary model and data files.

Important: When opening the notebooks (e.g., in VS Code or Jupyter Lab), ensure you select your activated .venv as the Python kernel.

### 2.1. Execution Order
Run the notebooks in the following order:

1. **EDA-and-Preprocessing.ipynb:** Downloads the AG News dataset, performs EDA, preprocesses, and tokenizes the data. This saves the prepared data to the ./data directory.

2. **Training-and-Fine-Tuning.ipynb:** Loads the preprocessed data and fine-tunes the DistilBERT model. This saves the final model to ./models/Distilbert/distilbert-final.

3. **Model-Evaluation.ipynb:** Evaluates the trained model on the test set.

## 3. Launch the Streamlit App

Once the model has been trained and saved, you can launch the application.

**Note:** Ensure your terminal is in the project root directory (AI-News-Classification/) and the virtual environment is active (.venv)
```bash
streamlit run app.py
```
The Streamlit application will automatically open in your web browser.