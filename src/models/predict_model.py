import pickle

import pandas as pd


def submit(path_to_model: str, path_to_testdata: str,
           path_to_submitdata: str):
    model = pickle.load(open(path_to_model, 'rb'))
    submit = pd.read_csv(path_to_submitdata)
    test_data = pd.read_parquet(path_to_testdata)
    pred = model.predict_proba(test_data)
    submit["prediction"] = pred[:, 1]
    submit.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    submit('models/test.pkl', 'data/processed/test.parquet',
           'data/raw/sample_submission.csv')
