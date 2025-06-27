# Fertilizer Predictor

This project predicts the most suitable fertilizer based on input features using machine learning models. It uses XGBoost for multi-class classification and provides model interpretability with LIME.

## Features
- Data preprocessing and cleaning
- Handling missing values
- Categorical encoding
- Model training with XGBoost
- Model evaluation using MAP@3
- Model interpretability with LIME
- Generates a submission file for predictions

## Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- dython
- lime

## Usage
1. Place the required datasets in the specified paths (see `main.py`).
2. Run the main script:
   ```bash
   python main.py
   ```
3. The script will output evaluation metrics and generate a `submission.csv` file with predictions.

## Project Structure
- `main.py` - Main script for data processing, model training, and prediction
- `README.md` - Project documentation

## Author
Keshav Bansal

## License
This project is licensed under the MIT License.
