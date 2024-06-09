# Google Stock Price Prediction using LSTM

This project demonstrates how to predict stock prices using Long Short-Term Memory (LSTM) neural networks with historical Google stock price data.

## Project Structure

- `data/`: Contains the dataset files.
  - `Google_train_data.csv`: Training dataset.
  - `Google_test_data.csv`: Testing dataset.
- `src/`: Contains the source code.
  - `stock_price_prediction.py`: Python script for data preprocessing, model training, and prediction.
- `models/`: (Optional) Contains saved models.
  - `lstm_model.h5`: Trained LSTM model.
- `README.md`: Project documentation.
- `requirements.txt`: Python dependencies required to run the project.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/vaibhavi1224/Stock_Price_Prediction.git
    cd Stock_Price_Prediction
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Navigate to the `src/` directory:
    ```bash
    cd src
    ```

2. Run the script:
    ```bash
    python stock_price_prediction.py
    ```

## Results

The script will output the following:

- Training loss plot
![image](https://github.com/vaibhavi1224/Stock_Price_Prediction/assets/149507676/bb7d5f30-f424-4338-be79-9240bed6d54d)

- Predicted vs. actual stock prices plot
  ![image](https://github.com/vaibhavi1224/Stock_Price_Prediction/assets/149507676/de35b12d-902b-4ccf-84db-2bf08fd73ea2)

## Contributing

Feel free to open issues or submit pull requests if you have any suggestions or improvements.

## License

This project is licensed under the MIT License.
