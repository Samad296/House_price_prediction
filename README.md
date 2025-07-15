# House Price Prediction API

A FastAPI-based API for predicting house prices using a pre-trained Linear Regression model.

## Description

This project provides a RESTful API to predict house prices based on features such as median income, house age, average rooms, population, and geographic coordinates. The model and scaler are loaded from pre-trained pickle files (`linear_regression.pkl` and `scaler.pkl`).

## Features

- Predict house prices using a trained Linear Regression model.
- Interactive API documentation via Swagger UI.
- Scalable and easy-to-deploy with FastAPI.

## Prerequisites

- Python 3.8+
- `pip` package manager
- Git (for cloning the repository)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Samad296/House_price_prediction.git
   cd House_price_prediction
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

4. Install the required dependencies:

   ```bash
   pip install fastapi uvicorn pydantic numpy pandas scikit-learn
   ```

5. Ensure `linear_regression.pkl` and `scaler.pkl` are in the project directory (generated from your training notebook).

## Usage

1. Run the FastAPI application:

   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

2. Open your browser and go to `http://127.0.0.1:8000/docs` to access the interactive API documentation.

3. Use the `/predict` endpoint to test predictions. Example POST request:

   ```json
   {
       "MedInc": 3.5,
       "HouseAge": 30.0,
       "AveRooms": 5.0,
       "AveBedrms": 1.0,
       "Population": 1000.0,
       "AveOccup": 2.5,
       "Latitude": 37.0,
       "Longitude": -122.0
   }
   ```

   Expected response:

   ```json
   {
       "prediction_price": <predicted_value>,
       "input_feature": {
           "MedInc": 3.5,
           "HouseAge": 30.0,
           "AveRooms": 5.0,
           "AveBedrms": 1.0,
           "Population": 1000.0,
           "AveOccup": 2.5,
           "Latitude": 37.0,
           "Longitude": -122.0
       }
   }
   ```

## Project Structure

- `main.py`: The FastAPI application code.
- `linear_regression.pkl`: Pre-trained model file.
- `scaler.pkl`: Pre-trained scaler file.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`.
3. Make your changes and commit: `git commit -m "Description of changes"`.
4. Push to the branch: `git push origin feature-branch`.
5. Open a pull request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- Built with FastAPI and scikit-learn.
- Inspired by machine learning deployment best practices.
