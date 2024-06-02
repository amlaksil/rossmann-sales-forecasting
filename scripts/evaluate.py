#!/usr/bin/python3

from src.model_evaluator import ModelEvaluator

def main():
    train_data_path = 'data/preprocessed_train_data.csv'
    model_path = 'best_model.pkl'
    evaluator = ModelEvaluator(train_data_path, model_path)
    evaluator.evaluate()

if __name__ == "__main__":
    main()
