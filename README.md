# Used Car Price Prediction

This project implements a complete end-to-end machine learning pipeline to predict used car prices. It encompasses data management, model training, evaluation, and deployment, while using several different tools to make it robust, reproducible, and scalable.

## Project Overview

This project consists of the following components:

*   **Data Acquisition:**  Data is loaded from a CSV file stored in an AWS S3 bucket, and versioned using DVC.
*   **Data Preprocessing:** An ETL process is used to clean and preprocess the data, and includes log transforms and feature engineering, to make the data ready for model training.
*   **Model Training & Evaluation:** Different machine learning models (including Linear Regression, Random Forest, Gradient Boosting, Decision Tree, K-Nearest Neighbors and XGBoost) are trained and evaluated, with results tracked using MLflow on DagsHub.
*   **Streamlit Application:** A user-friendly web application built with Streamlit allows users to input car details and receive price predictions from the trained model.
*   **Deployment:** The Streamlit application is containerized using Docker, and is deployed on AWS EC2 to make it accessible on the internet.
*  **Data Version Control**: DVC is used to version control the data, to make the project reproducible and track changes to the data.
*  **Model Tracking**: MLflow is used to track the training experiments and to save the trained models.

## Technologies Used

*   **Python:** Main programming language.
*   **Pandas & NumPy:** Data manipulation and numerical computations.
*   **Scikit-learn:** Machine learning algorithms and tools.
*   **Streamlit:** Building the web application.
*   **MLflow:** Experiment tracking and model management.
*  **DVC (Data Version Control):** Data versioning and pipeline management.
*   **Docker:** Containerization.
*   **AWS EC2 & S3:** Cloud deployment and storage.
*   **boto3:** For connecting to AWS S3.
*   **python-dotenv:** For managing environment variables.
*   **XGBoost:** Gradient Boosting Framework
*   **Requests:** HTTP library
*   **Selenium**: Used for web scraping (but not used in the final implementation)

## Setup and Installation

To set up this project, follow these steps:

1.  **Clone the Repository:** Clone this repository to your local machine using Git:
    ```bash
    git clone <https://github.com/vyshnev/used-car-price-prediction.git>
    cd used-car-price-prediction
    ```
2.  **Create a Virtual Environment:** Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Linux and MacOS
    venv\Scripts\activate      # On Windows
    ```
3.  **Install Dependencies:** Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure AWS Credentials:**
    * Set up your AWS credentials using the AWS CLI. Make sure you have the correct credentials to access your bucket, which we configured in step 3 of the project.
5. **Configure DagsHub Credentials:**
    * Set up a DagsHub account, and configure your credentials and your tracking URI in the `.env` file in the project's root directory. Make sure that your password or personal token is configured correctly.
6.   **Configure AWS S3 Bucket Name:** In the `.env` file, include your AWS S3 bucket name using the variable `AWS_S3_BUCKET_NAME=used-car-price-prediction-vyshnev`.

## Running the Application

1.  **Run the Training Script** To train the models, run the following command:

    ```bash
    python src/model/train_model.py
    ```
    This will train the defined models, track metrics and save your models in your DagsHub account.

2.  **Run the Streamlit Application** To run the Streamlit application, use the following command:

    ```bash
    streamlit run app/app.py
    ```
    Your application will be accessible in your browser through the address specified in the console.

3. **Dockerize the application:**
 * To create your docker image, run the following command:
    ```bash
      docker build -t used-car-price-prediction .
    ```
 * To run the dockerized application:
    ```bash
      docker run -p 8501:8501 used-car-price-prediction
    ```
 * This command will start the application in a container in the specified port, and it can be tested in a browser.

4. **Deploy to AWS EC2**:
* See the section "[Deploying the Containerized Streamlit App on AWS EC2](DEPLOYMENT.md)" to deploy the application in the cloud.

## Project Structure
```bash
used-car-price-prediction/
├── data/ # To hold raw data, processed data
│ └── raw/
│ └── cars24-used-cars-dataset.csv
├── notebooks/ # Jupyter notebooks for EDA, experiments
├── src/ # Python source code
│ ├── init.py # Make src a package
│ ├── etl/ # ETL related files
│ │ └── data_scraping.py # Data loading and preprocessing
│ └── models/ # Model related files
│ └── preprocess.py # Data preprocessing
│ └── train_model.py # Model Training
├── test/ # Unit and integration tests
│ └──test_model.py # Testing code
├── app/ # For Streamlit application
│ └── app.py # Streamlit app
├── requirements.txt # Python dependencies
├── .gitignore
├── Dockerfile # Dockerfile for the app
└── README.md # Project documentation
```

## Contributing

This was a challenging project and your contribution and feedback is welcome to improve it. If you find any issue, please open an issue in the project's github repository.
