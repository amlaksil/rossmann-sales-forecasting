#!/usr/bin/python3

from src.data_preprocessor import DataPreprocessor

def main():
    train_data_path = 'data/train.csv'
    test_data_path = 'data/test.csv'
    store_data_path = 'data/store.csv'
    output_train_path = 'data/preprocessed_train_data.csv'
    output_test_path = 'data/preprocessed_test_data.csv'
    preprocessor = DataPreprocessor(train_data_path, test_data_path, store_data_path, output_train_path, output_test_path)
    preprocessor.run()

if __name__ == "__main__":
    main()
