#!/usr/bin/python3
from src.model_trainer import ModelTrainer


def main(save_model=False):
    train_data_path = 'data/preprocessed_train_data.csv'
    trainer = ModelTrainer(train_data_path)
    trainer.train()

    if save_model:
        trainer.save_model('best_model.pkl')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-model', action='store_true', help='Save the trained model')
    args = parser.parse_args()
    main(args.save_model)
