# StockPrice_Prediction
 Stock prices prediction [Time series forecasting] using the AAPL dataset using Python.

---

# Stock Price Prediction Using LSTM

This project is a time-series forecasting model that predicts stock prices using a Long Short-Term Memory (LSTM) neural network. The model is built using historical stock price data of Apple Inc. (AAPL) and aims to predict future stock prices based on past trends.

## Project Overview

The project focuses on the following tasks:
1. **Data Preprocessing**: Loading and preparing the stock price data, which includes:
   - Resampling the data to ensure daily frequency.
   - Creating a target column (`Future_Price`) representing future stock prices.
   - Normalizing the features for better performance during model training.
2. **Model Building**: Building and training an LSTM model using `TensorFlow` and `Keras` to predict future stock prices.
3. **Model Evaluation**: Evaluating the model performance by calculating the Mean Absolute Error (MAE) on the test set.
4. **Predicting Future Prices**: Using the trained model to predict stock prices for the next 30 days.

## Dataset

The dataset used in this project is historical stock data for Apple Inc. (AAPL), which contains the following columns:
- `Date`: The date of the stock price.
- `Open`: Opening price of the stock.
- `High`: Highest price of the stock on that date.
- `Low`: Lowest price of the stock on that date.
- `Close`: Closing price of the stock on that date.
- `Volume`: Volume of stock traded.

The data was loaded from a CSV file (`AAPL.csv`), and further processing was applied to prepare it for the LSTM model.

**Dataset link** : https://www.kaggle.com/datasets/meetnagadia/apple-stock-price-from-19802021

## Project Structure

The code is divided into the following sections:

- **Data Loading and Preprocessing**: Load the dataset, set the datetime index, resample to daily frequency, and prepare the target variable (`Future_Price`).
- **Data Splitting**: Split the data into training and testing sets using an 80-20 split.
- **Normalization**: Normalize the features using `MinMaxScaler` to improve model training.
- **Model Creation**: Build an LSTM model with `50` units and compile it using the Adam optimizer and Mean Squared Error (MSE) loss function.
- **Model Training**: Train the model on the training set with `50` epochs and a batch size of `1`.
- **Prediction and Evaluation**: Predict stock prices on the test set and evaluate the model using Mean Absolute Error (MAE).
- **Future Price Prediction**: Predict future stock prices for the next `30` days based on recent data.

## Installation

To run this project, install the required libraries using:

```bash
pip install pandas numpy scikit-learn tensorflow keras
```

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Atchuth01/StockPrice_Prediction.git
   ```

2. **Run the script**:
   ```bash
   python StockPrices[TimeSeries_forecasting].py
   ```

3. **Results**:
   - The predicted future stock prices will be displayed in the console.
   - Model performance (MAE) will also be shown for the test set.

## Future Work

- Experiment with different model architectures and hyperparameters to improve prediction accuracy.
- Incorporate additional features (such as technical indicators or financial news sentiment) to enhance the model.
- Implement real-time stock data for live predictions.

## License

This project is licensed under the MIT License.

## Author
Author : Atchuth V