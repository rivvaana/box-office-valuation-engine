# Box Office Valuation Engine

An end-to-end localized analytics dashboard that transitions a deep learning regression model out of a Jupyter Notebook into a functional user interface. 

### Core Features
* **Predictive Backend:** Powered by a multi-layer Artificial Neural Network (ANN) built using TensorFlow and Keras to forecast global film revenue.
* **Real-Time Data Pipeline:** Features an embedded standard scaling pipeline within the script execution to normalize live inputs (budget, audience engagement, popularity index) against baseline training data variance.
* **Streamlit UI Customization:** Designed with a single-view, scroll-free parallel layout using custom CSS injections to maximize dashboard efficiency.

### How to Run Locally
1. Clone or download this repository.
2. Install dependencies: `pip install streamlit tensorflow numpy`
3. Run the application: `streamlit run app.py`
