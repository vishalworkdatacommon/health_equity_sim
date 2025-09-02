---
title: Health Equity & Policy Simulation Tool
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8501
---
# Health Equity & Policy Simulation Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive, industry-grade web application for predicting and understanding health insurance coverage disparities in the United States using machine learning, deep learning, and MLOps.

## About The Project

This project provides a web-based dashboard for policymakers, researchers, and public health officials to simulate health insurance coverage rates based on various demographic and geographic factors. It leverages the Small Area Health Insurance Estimates (SAHIE) dataset from the U.S. Census Bureau.

The tool is built with a focus on modern MLOps practices, including experiment tracking, model explainability, and containerization, making it a robust and deployable solution.

### Key Features

*   **Dual Model Prediction:** Get predictions from both a high-performance LightGBM model and a TensorFlow/Keras deep learning model.
*   **Model Explainability:** Understand *why* a prediction was made using SHAP (SHapley Additive exPlanations) visualizations for each simulation.
*   **Experiment Tracking:** All model training runs are logged with MLflow, capturing parameters, metrics, and model artifacts for full reproducibility.
*   **Containerized Application:** The entire Streamlit application is containerized with Docker, ensuring consistent and easy deployment.
*   **Data Processing Pipeline:** Includes scripts for cleaning, feature-engineering, and preparing the raw SAHIE data for analysis.

## How to Use This Space

1.  Use the sidebar on the left to select the **Year, State, County, and Demographic** parameters for your simulation.
2.  Click the **"Run Simulation"** button.
3.  View the predicted uninsured rates from both the LightGBM and Deep Learning models.
4.  Analyze the **SHAP Force Plot** to understand which factors contributed most to the prediction.

## Local Setup & Usage

Follow these steps to set up and run the project locally.

### Prerequisites

*   Python 3.9+
*   Docker (for running the containerized application)
*   An environment manager like `venv` or `conda`.

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone <your-repo-url>
    cd health-equity-sim
    ```

2.  **Create a virtual environment and install dependencies:**
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Download Raw Data:**
    This project requires the raw SAHIE CSV files. Place them inside the `input_files/` directory.

### Workflow

1.  **Process the Data:**
    ```sh
    python3 feature_engineering.py
    ```
2.  **Train the Models:**
    ```sh
    python3 model.py
    python3 deep_learning.py
    ```
3.  **Run the Streamlit App:**
    ```sh
    streamlit run app.py
    ```