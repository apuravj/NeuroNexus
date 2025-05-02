# Main script to run the pipeline

from src.preprocess import load_and_preprocess_data
from src.model import split_data, handle_imbalance, train_and_evaluate

def main():
    data_path = "data/creditcard.csv"
    df = load_and_preprocess_data(data_path)
    
    X_train, X_test, y_train, y_test = split_data(df)
    X_resampled, y_resampled = handle_imbalance(X_train, y_train)
    
    train_and_evaluate(X_resampled, X_test, y_resampled, y_test)

if __name__ == "__main__":
    main()
