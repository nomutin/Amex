from typing import List

import lightgbm as lgbm
import numpy as np
import optuna.integration.lightgbm as opt_lgb
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split


def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    if isinstance(y_true, np.ndarray):
        y_true = pd.DataFrame(y_true, columns=["target"])

    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred, columns=["prediction"])
        #y_pred["prediction"] = y_pred

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))

        df['weight'] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df["target"] == 1).sum()

    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df["target"] * df['weight']).sum()
        df['cum_pos_found'] = (df["target"] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    d = top_four_percent_captured(y_true, y_pred)
    g = normalized_weighted_gini(y_true, y_pred)

    return 0.5 * (g + d)


def lgb_amex_metric(y_true, y_pred):
    """The competition metric with lightgbm's calling convention"""
    return ('amex', amex_metric(y_true, y_pred), True)


def main(processed_data_path: List[str],
         path_to_testdata: str,
         path_to_submitdata: str
         ) -> None:
    input_df = pd.read_parquet(processed_data_path[0])
    target_df = pd.read_parquet(processed_data_path[1])
    submit_data = pd.read_csv(path_to_submitdata)
    test_data = pd.read_parquet(path_to_testdata)

    params = {
        'objective': 'binary',
        'random_seed': 9999,
        'metric': 'binary_logloss',
        'feature_pre_filter': False,
        'lambda_l1': 0.008159183626572277,
        'lambda_l2': 0.42574294825098996,
        'num_leaves': 250,
        'feature_fraction': 0.88,
        'bagging_fraction': 0.8220994451463124,
        'bagging_freq': 7,
        'min_child_samples': 25,
        'num_iterations': 500,
    }

    kf = KFold(n_splits=3)
    models = []
    for train_index, val_index in kf.split(input_df):
        X_train = input_df.iloc[train_index]
        X_valid = input_df.iloc[val_index]
        Y_train = target_df.iloc[train_index]
        Y_valid = target_df.iloc[val_index]
        train_dataset = lgbm.Dataset(X_train, Y_train)
        val_dataset = lgbm.Dataset(X_valid, Y_valid, reference=train_dataset)

        callbacks = [
            lgbm.log_evaluation(1),
            lgbm.early_stopping(stopping_rounds=10, verbose=True),
        ]
        model = lgbm.train(params,
                           train_dataset,
                           valid_sets=val_dataset,
                           num_boost_round=2000,
                           callbacks=callbacks,
                           verbose_eval=10,
                           )
        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        print(accuracy_score(Y_valid, np.round(y_pred)))
        models.append(model)

        preds = []
        for model in models:
            pred = model.predict(test_data)
            preds.append(pred)

        preds = np.mean(np.array(preds), axis=0)
        submit_data["prediction"] = preds
        submit_data.to_csv('submission.csv', index=False)


def serarch_hparams(processed_data_path: List[str]):
    input_df = pd.read_parquet(processed_data_path[0])
    target_df = pd.read_parquet(processed_data_path[1])

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'random_seed': 9999
    }
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        input_df, target_df, test_size=0.2, shuffle=True, random_state=42)
    train_dataset = lgbm.Dataset(X_train, Y_train)
    val_dataset = lgbm.Dataset(X_valid, Y_valid, reference=train_dataset)

    callbacks = [
        lgbm.early_stopping(stopping_rounds=5, verbose=True),
        lgbm.log_evaluation(1)
    ]
    model = opt_lgb.train(params,
                          train_dataset,
                          valid_sets=val_dataset,
                          num_boost_round=500,
                          callbacks=callbacks,
                          verbose_eval=10,
                          )
    print(model.params)


if __name__ == '__main__':
    main(
        ['data/processed/inputs.parquet', 'data/processed/targets.parquet'],
        path_to_testdata='data/processed/test.parquet',
        path_to_submitdata='data/raw/sample_submission.csv'
    )
    # serarch_hparams(
    #     ['data/processed/inputs.parquet', 'data/processed/targets.parquet'],
    # )
    #  {'objective': 'binary', 'metric': 'binary_logloss', 'feature_pre_filter': False, 'lambda_l1': 0.008159183626572277, 'lambda_l2': 0.42574294825098996, 'num_leaves': 250, 'feature_fraction': 0.88, 'bagging_fraction': 0.8220994451463124, 'bagging_freq': 7, 'min_child_samples': 25, 'num_iterations': 500, 'early_stopping_round': None}
