import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocessing(df_):
    id_job_ = df_['id_job']

    df_ = pd.DataFrame(np.array([np.array(x.split(','), dtype='int') for x in list(df_['features'])]))

    feature_type_ = df_[0].loc[0]
    df_ = df_.drop([0], axis=1)
    df_.columns = [f'feature_{feature_type_}_{"{" + str(i - 1) + "}"}' for i in df_.columns]

    return df_, feature_type_, id_job_


def main():
    df_train = pd.read_csv('train.tsv', '\t')
    df_test = pd.read_csv('test.tsv', '\t')

    df_train_features, _, _ = preprocessing(df_train)
    df_test_features, feature_type, id_job_test = preprocessing(df_test)

    scaler = StandardScaler()
    scaler.fit(df_train_features)

    res_df = pd.DataFrame()

    res_df[f'max_feature_{feature_type}_index'] = df_test_features.idxmax(axis=1)
    res_df[f'max_feature_{feature_type}_index'] = res_df[f'max_feature_{feature_type}_index'].apply(
        lambda s: (s.split('{'))[1].split('}')[0])

    max_feature_test = df_test_features.max(axis=1)
    mean_train = df_train_features.mean(axis=0)

    res_df[f'max_feature_{feature_type}_abs_mean_diff'] = list(
        map(lambda index, max_: max_ - mean_train[f'feature_{feature_type}_{"{" + index + "}"}'],
            res_df[f'max_feature_{feature_type}_index'], max_feature_test))

    df_test_features = pd.DataFrame(scaler.transform(df_test_features),
                                    columns=[col.replace(f'feature_{feature_type}_', f'feature_{feature_type}_stand_')
                                             for col in df_test_features.columns])

    df_test_features['id_job'] = id_job_test
    df_test_features[f'max_feature_{feature_type}_index'] = res_df[f'max_feature_{feature_type}_index']
    df_test_features[f'max_feature_{feature_type}_abs_mean_diff'] = res_df[f'max_feature_{feature_type}_abs_mean_diff']

    df_test_features = df_test_features[
        ['id_job'] + [c for c in df_test_features if c not in ['id_job']]]

    df_test_features.to_csv('test_proc.tsv', index=0)


if __name__ == '__main__':
    main()
