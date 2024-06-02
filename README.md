# Rossmann Sales Forecasting

This project aims to forecast sales for Rossmann Pharmaceuticals stores across several cities, six weeks ahead of time. The prediction model considers various factors including promotions, competition, holidays, seasonality, and locality. The end-to-end solution includes data preprocessing, model training, hyperparameter tuning, and deployment.

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Model Deployment](#model-deployment)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
Managers in individual stores rely on their years of experience and personal judgment to forecast sales. This project builds a machine learning model to provide accurate sales forecasts, assisting the finance team in planning and decision-making.

## Data
The dataset for this project is provided by Kaggle and includes various features related to store operations and external factors:
- Store, Sales, Customers, Open, StateHoliday, SchoolHoliday
- StoreType, Assortment, CompetitionDistance, CompetitionOpenSince, Promo, Promo2, Promo2Since, PromoInterval

The data can be downloaded from [Kaggle Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data).

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/rossmann-sales-forecasting.git
    cd rossmann-sales-forecasting
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the dataset from Kaggle and place it in the `data/` directory.

## Usage
1. **Data Preprocessing:**
    ```bash
    python scripts/preprocess.py
    ```

2. **Model Training and Hyperparameter Tuning:**
    ```bash
    python scripts/train.py
    ```

3. **Evaluate the Model:**
    ```bash
    python scripts/evaluate.py
    ```

## Model Deployment
1. **Save the Trained Model:**
    ```bash
    python scripts/train.py --save-model
    ```

2. **Run the Flask API:**
    ```bash
    python scripts/app.py
    ```

3. **Make Predictions:**
    Send a POST request to the `/predict` endpoint with the required input data.

    Example:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"Store": 1, "DayOfWeek": 5, "Customers": 555, ...}' http://127.0.0.1:5000/predict
    ```

## Results
The model's performance is evaluated using RMSE. The current best RMSE is:

| Model                | RMSE   |
|----------------------|--------|
| RandomForestRegressor| 24.29  |
| Best Model           | Y.YY   |

## Contributing
Contributions are welcome! Please create an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
