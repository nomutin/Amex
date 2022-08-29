from typing import List

import pandas as pd


def process_train(input_filepath: List[str], output_filepath: List[str]):
    train_data = pd.read_parquet(input_filepath[0]).set_index(
        'customer_ID', drop=True).sort_index()

    labels = pd.read_csv(input_filepath[1])\
        .set_index('customer_ID', drop=True).sort_index()

    df = pd.merge(train_data, labels, left_index=True, right_index=True)
    categorical_cols = ['S_2', 'B_30', 'B_38', 'D_114', 'D_116', 'D_117',
                        'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68',
                        'target']
    features = [c for c in df.columns if c not in categorical_cols]
    inputs = df[features]
    targets = df[['target']]
    inputs.to_parquet(output_filepath[0])
    targets.to_parquet(output_filepath[1])


def process_test(input_filepath: List[str], output_filepath: List[str]):
    test_data = pd.read_parquet(input_filepath[0]).groupby('customer_ID')\
        .tail(1).set_index('customer_ID', drop=True).sort_index()
    categorical_cols = ['S_2', 'B_30', 'B_38', 'D_114', 'D_116', 'D_117',
                        'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    features = [c for c in test_data.columns if c not in categorical_cols]
    test_data = test_data[features]
    test_data.to_parquet(output_filepath[0])


if __name__ == '__main__':
    process_train(input_filepath=['data/raw/train.parquet',
                                  'data/raw/train_labels.csv'],
                  output_filepath=['data/processed/inputs.parquet',
                                   'data/processed/targets.parquet']
                  )
    # process_test(['data/raw/test.parquet'], ['data/processed/test.parquet'])
