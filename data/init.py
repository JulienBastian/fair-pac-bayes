from pathlib import Path
import os
import torch
import numpy as np
import pandas as pd
from folktables import (
    ACSDataSource,
    ACSIncome,
    ACSMobility,
    ACSTravelTime,
)
from aif360.sklearn.datasets import fetch_compas, fetch_lawschool_gpa, fetch_german, fetch_meps
from sklearn.model_selection import train_test_split

BINARY_DATASETS = (
    'ACSINC',
    'ACSMOB',
    'ACSTRAVEL',
    'COMPAS',
    'GERMAN',
    'LAW',
    'MEPS',
)


def _resolve_offline_flag(offline):
    """
    Resolve offline mode.
    If offline is None, use env var PBF_OFFLINE (1/true/yes -> True).
    """
    if offline is not None:
        return bool(offline)
    return os.getenv("PBF_OFFLINE", "0").strip().lower() in {"1", "true", "yes"}


def _load_cached_table(base_path):
    """
    Load a cached table from <base>.parquet or <base>.pkl.
    Returns DataFrame or None if missing.
    """
    parquet = Path(str(base_path) + ".parquet")
    pkl = Path(str(base_path) + ".pkl")
    if parquet.is_file():
        return pd.read_parquet(parquet)
    if pkl.is_file():
        return pd.read_pickle(pkl)
    return None


def _save_cached_table(df, base_path):
    """
    Save DataFrame to parquet (preferred) or pickle (fallback).
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    parquet = Path(str(base_path) + ".parquet")
    pkl = Path(str(base_path) + ".pkl")
    try:
        df.to_parquet(parquet, index=False)
        return parquet
    except Exception:
        df.to_pickle(pkl)
        return pkl


def _load_cached_xy(raw_root, name):
    """
    Load X/y pair from data/raw/aif360/<name>_X.(parquet|pkl) + <name>_y.npy
    Returns (X_df, y_np) or (None, None).
    """
    x_df = _load_cached_table(raw_root / "aif360" / f"{name}_X")
    y_path = raw_root / "aif360" / f"{name}_y.npy"
    if x_df is None or not y_path.is_file():
        return None, None
    return x_df, np.load(y_path, allow_pickle=True)


def _get_sensitive_column_index(dataset, feature_names, sensible_variable):
    """
    Return the sensitive-column index by matching its feature name.
    This remains robust to column reordering.
    """
    sensible_map = {
        'ACSINC': {
            'GENDER': 'SEX',
            'OTHER': 'RAC1P'
        },
        'COMPAS': {
            'GENDER': 'sex',
            'OTHER': 'race'
        },
        'GERMAN': {
            'GENDER': 'sex',
            'OTHER': 'age_threshold'
        },
        'LAW': {
            'GENDER': 'gender',
            'OTHER': 'race'
        },
        'MEPS': {
            'GENDER': 'SEX',
            'OTHER': 'RACE'
        }
    }
    
    if dataset not in sensible_map:
        raise NotImplementedError(f"Dataset {dataset} not in sensible_map")
    
    if sensible_variable not in sensible_map[dataset]:
        raise NotImplementedError(f"Sensitive variable {sensible_variable} not supported for {dataset}")
    
    feature_name = sensible_map[dataset][sensible_variable]
    
    try:
        idx = list(feature_names).index(feature_name)
        print(f"✓ Found {dataset} - {sensible_variable}: '{feature_name}' at index {idx}")
        return idx
    except ValueError:
        raise ValueError(f"Feature '{feature_name}' not found in feature_names: {feature_names}")

def fetch_data(dataset, sensible_variable, pred, test_size=0.2, seed=None, data_root="./data", offline=None):
    """
    Load raw data with local-first strategy:
    - if cached raw file exists in data/raw, load it
    - otherwise, fetch from external source (unless offline=True)
    Then apply the same preprocessing as before.
    """
    offline = _resolve_offline_flag(offline)
    data_root = Path(data_root)
    raw_root = data_root / "raw"

    if dataset == "ACSINC":
         acs_base = raw_root / "acs" / "acs_2018_1Y_person_AL_AZ"
         acs_data = _load_cached_table(acs_base)
         if acs_data is None:
             if offline:
                 raise FileNotFoundError(
                     f"[OFFLINE] Missing local raw ACS file. Expected {acs_base}.parquet or .pkl"
                 )
             data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
             acs_data = data_source.get_data(states=['AL', 'AZ'], download=True)
             saved = _save_cached_table(acs_data, acs_base)
             print(f"[ACSINC] cached raw table at: {saved}")
         feature_names = ACSIncome.features
         X, label, group_race = ACSIncome.df_to_numpy(acs_data)
         Y = np.where(label == 1, 1, -1)
         sv_col = _get_sensitive_column_index('ACSINC', feature_names, sensible_variable)
         s = X[:, sv_col].copy()
         if sensible_variable == 'GENDER':
             s = (s == 1).astype(int)   # Male vs Female
         else:
             s = (s == 1).astype(int)   # White vs Non-white (RAC1P==1)

         for feat in ['SEX', 'RAC1P']:
             col = feature_names.index(feat)
             X[:, col] = (X[:, col] == 1).astype(int)

         # Remove original sensitive column so we can reinsert it cleanly in column 0.
         X = np.delete(X, sv_col, 1)
         feature_names_wo_sv = [f for j, f in enumerate(feature_names) if j != sv_col]

         col_to_remove = [feature_names_wo_sv.index("OCCP"), feature_names_wo_sv.index("POBP")]
         for feat in ['COW', 'MAR', 'RELP']:
             col = feature_names_wo_sv.index(feat)
             col_to_remove.append(col)
             one_hot = np.eye(np.max(X[:, col]).astype(int))[(X[:, col]-1).astype(int)]
             X = np.hstack((X, one_hot))
         X = np.delete(X, col_to_remove, 1)
         X = np.hstack((s[:, None], X))
         sv_col = 0

    elif dataset in ('ACSMOB', 'ACSTRAVEL'):
         acs_base = raw_root / "acs" / "acs_2018_1Y_person_AL_AZ"
         acs_data = _load_cached_table(acs_base)
         if acs_data is None:
             if offline:
                 raise FileNotFoundError(
                     f"[OFFLINE] Missing local raw ACS file. Expected {acs_base}.parquet or .pkl"
                 )
             data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
             acs_data = data_source.get_data(states=['AL', 'AZ'], download=True)
             saved = _save_cached_table(acs_data, acs_base)
             print(f"[{dataset}] cached raw table at: {saved}")

         if dataset == 'ACSMOB':
             problem = ACSMobility
         else:
             problem = ACSTravelTime

         feature_names = list(problem.features)
         X, label, _ = problem.df_to_numpy(acs_data)
         if dataset == 'ACSTRAVEL':
             Y = np.where(label == 1, -1, 1)
         else:
             Y = np.where(label == 1, 1, -1)

         if sensible_variable == 'GENDER':
             sensitive_feature = 'SEX'
         elif sensible_variable == 'OTHER':
             sensitive_feature = 'RAC1P'
         else:
             raise NotImplementedError(
                 f"Sensitive variable {sensible_variable} not supported for {dataset}"
             )

         sv_col = feature_names.index(sensitive_feature)
         s = (X[:, sv_col] == 1).astype(int)
         X = np.delete(X, sv_col, axis=1)
         X = np.hstack((s[:, None], X))
         sv_col = 0

    elif dataset == 'COMPAS':
         x_df, y_raw = _load_cached_xy(raw_root, "compas")
         if x_df is None:
             if offline:
                 raise FileNotFoundError(
                     "[OFFLINE] Missing local COMPAS raw files. "
                     "Expected data/raw/aif360/compas_X.(parquet|pkl) and compas_y.npy"
                 )
             data = fetch_compas(binary_race=True, numeric_only=True)
             x_df = data.X
             y_raw = data.y.to_numpy()
             saved = _save_cached_table(x_df, raw_root / "aif360" / "compas_X")
             np.save(raw_root / "aif360" / "compas_y.npy", y_raw)
             print(f"[COMPAS] cached raw table at: {saved}")
         X, Y = x_df.to_numpy(), y_raw
         Y = np.abs(Y - 1)
         feature_names = list(x_df.columns)
         sv_col = _get_sensitive_column_index('COMPAS', feature_names, sensible_variable)


    elif dataset == 'GERMAN':
         x_df, y_raw = _load_cached_xy(raw_root, "german")
         if x_df is None:
             if offline:
                 raise FileNotFoundError(
                     "[OFFLINE] Missing local GERMAN raw files. "
                     "Expected data/raw/aif360/german_X.(parquet|pkl) and german_y.npy"
                 )
             data = fetch_german(numeric_only=True, binary_age=True)
             x_df = data.X
             y_raw = data.y.to_numpy()
             saved = _save_cached_table(x_df, raw_root / "aif360" / "german_X")
             np.save(raw_root / "aif360" / "german_y.npy", y_raw)
             print(f"[GERMAN] cached raw table at: {saved}")
         x_df['age_threshold'] = (x_df.age >= 25).astype(int)
         X, Y = x_df.to_numpy(), y_raw
         feature_names = list(x_df.columns)
         sv_col = _get_sensitive_column_index('GERMAN', feature_names, sensible_variable)

    elif dataset == 'LAW':
         x_df, y_raw = _load_cached_xy(raw_root, "law")
         if x_df is None:
             if offline:
                 raise FileNotFoundError(
                     "[OFFLINE] Missing local LAW raw files. "
                     "Expected data/raw/aif360/law_X.(parquet|pkl) and law_y.npy"
                 )
             data = fetch_lawschool_gpa(numeric_only=True)
             x_df = data.X
             y_raw = data.y.to_numpy()
             saved = _save_cached_table(x_df, raw_root / "aif360" / "law_X")
             np.save(raw_root / "aif360" / "law_y.npy", y_raw)
             print(f"[LAW] cached raw table at: {saved}")
         X, Y = x_df.to_numpy(), y_raw
         Y = (Y >= np.median(Y)).astype(int)
         feature_names = list(x_df.columns)
         sv_col = _get_sensitive_column_index('LAW', feature_names, sensible_variable)


    elif dataset == 'MEPS':
         x_df, y_raw = _load_cached_xy(raw_root, "meps")
         if x_df is None:
             if offline:
                 raise FileNotFoundError(
                     "[OFFLINE] Missing local MEPS raw files. "
                     "Expected data/raw/aif360/meps_X.(parquet|pkl) and meps_y.npy"
                 )
             data = fetch_meps(panel=19, accept_terms=True)
             x_df = data.X
             y_raw = data.y.to_numpy()
             saved = _save_cached_table(x_df, raw_root / "aif360" / "meps_X")
             np.save(raw_root / "aif360" / "meps_y.npy", y_raw)
             print(f"[MEPS] cached raw table at: {saved}")
         x_df.SEX = (x_df.SEX == 'Male').astype(int)
         x_df.RACE = (x_df.RACE == 'White').astype(int)
         X = x_df.to_numpy()
         Y = (y_raw == "< 10 Visits").astype(int)
         feature_names = list(x_df.columns)
         sv_col = _get_sensitive_column_index('MEPS', feature_names, sensible_variable)


    else:
        raise NotImplementedError("Dataset not supported")

    print(f"The dataset contains {len(Y)} examples.")
    Y[Y == 0] = -1

    # Force sensitive attribute to 0/1 when encoded as -1/+1.
    s_values = np.unique(X[:, sv_col])
    if len(s_values) == 2 and np.all(np.isin(s_values, [-1, 1])):
        X[:, sv_col] = ((X[:, sv_col] + 1) / 2).astype(int)
    elif not np.all(np.isin(s_values, [0, 1])):
        raise ValueError(
            f"Sensitive attribute must be binary in {{0,1}} or {{-1,1}}; got {s_values}."
        )

    # Move the sensitive feature to the first column.
    if sv_col != 0:
        X[:, [0, sv_col]] = X[:, [sv_col, 0]]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)
    if pred == "RandomForests":
        joint_labels = np.column_stack(
            (y_train, X_train[:, 0])
        )
        _, strata = np.unique(
            joint_labels,
            axis=0,
            return_inverse=True,
        )
        (
            X_train_prior,
            X_train,
            y_train_prior,
            y_train,
        ) = train_test_split(
            X_train,
            y_train,
            test_size=0.5,
            random_state=seed,
            stratify=strata,
        )
    else:
        X_train_prior, y_train_prior = None, None
    X_train0 = X_train[y_train == -1]
    X_train1 = X_train[y_train ==  1]
    X_test0 = X_test[y_test == -1]
    X_test1 = X_test[y_test ==  1]
    y_train0 = y_train[y_train == -1]
    y_train1 = y_train[y_train ==  1]
    y_test0 = y_test[y_test == -1]
    y_test1 = y_test[y_test ==  1]

    X = {'train': X_train,
         'train_prior': X_train_prior,
         'train_A': X_train[X_train[:, 0] == np.unique(X_train[:, 0])[0]],
         'train_B': X_train[X_train[:, 0] == np.unique(X_train[:, 0])[1]],
         'train_A0': X_train0[X_train0[:, 0] == np.unique(X_train0[:, 0])[0]],
         'train_B0': X_train0[X_train0[:, 0] == np.unique(X_train0[:, 0])[1]],
         'train_A1': X_train1[X_train1[:, 0] == np.unique(X_train1[:, 0])[0]],
         'train_B1': X_train1[X_train1[:, 0] == np.unique(X_train1[:, 0])[1]],
         'train_0': X_train0,
         'train_1': X_train1,

         'test': X_test,
         'test_A': X_test[X_test[:, 0] == np.unique(X_test[:, 0])[0]],
         'test_B': X_test[X_test[:, 0] == np.unique(X_test[:, 0])[1]],
         'test_A0': X_test0[X_test0[:, 0] == np.unique(X_test0[:, 0])[0]],
         'test_B0': X_test0[X_test0[:, 0] == np.unique(X_test0[:, 0])[1]],
         'test_A1': X_test1[X_test1[:, 0] == np.unique(X_test1[:, 0])[0]],
         'test_B1': X_test1[X_test1[:, 0] == np.unique(X_test1[:, 0])[1]],
         'test_0': X_test0,
         'test_1': X_test1}

    y = {'train': y_train,
         'train_prior': y_train_prior,
         'train_A': y_train[X_train[:, 0] == np.unique(X_train[:, 0])[0]],
         'train_B': y_train[X_train[:, 0] == np.unique(X_train[:, 0])[1]],
         'train_A0': y_train0[X_train0[:, 0] == np.unique(X_train0[:, 0])[0]],
         'train_B0': y_train0[X_train0[:, 0] == np.unique(X_train0[:, 0])[1]],
         'train_A1': y_train1[X_train1[:, 0] == np.unique(X_train1[:, 0])[0]],
         'train_B1': y_train1[X_train1[:, 0] == np.unique(X_train1[:, 0])[1]],
         'train_0': y_train0,
         'train_1': y_train1,

         'test': y_test,
         'test_A': y_test[X_test[:, 0] == np.unique(X_test[:, 0])[0]],
         'test_B': y_test[X_test[:, 0] == np.unique(X_test[:, 0])[1]],
         'test_A0': y_test0[X_test0[:, 0] == np.unique(X_test0[:, 0])[0]],
         'test_B0': y_test0[X_test0[:, 0] == np.unique(X_test0[:, 0])[1]],
         'test_A1': y_test1[X_test1[:, 0] == np.unique(X_test1[:, 0])[0]],
         'test_B1': y_test1[X_test1[:, 0] == np.unique(X_test1[:, 0])[1]],
         'test_0': y_test0,
         'test_1': y_test1}
    
    s = {
        key: None if value is None else value[:, 0] for key, value in X.items()
    }

    return dict(X=X, y=y, s=s)


class Dataset:
    """
    Code adapted from https://github.com/Qwicen/node/blob/master/lib/data.py .
    """

    def __init__(self, dataset, sensible_variable, pred, data_path='./data', normalize=False, seed=None, offline=None):
        """
        Dataset is a dataclass that contains all training and evaluation data required for an experiment
        :param dataset: a pre-defined dataset name (see DATASETS) or a custom dataset
            Your dataset should be at (or will be downloaded into) {data_path}/{dataset}
        :param data_path: a shared data folder path where the dataset is stored (or will be downloaded into)
        :param normalize: standardize features by removing the mean and scaling to unit variance
        """

        path = Path(data_path)
        path.mkdir(parents=True, exist_ok=True)

        if dataset not in BINARY_DATASETS:
            raise NotImplementedError("Dataset not supported")
        self.binary = True

        data_dict = fetch_data(dataset, sensible_variable, pred, test_size=0.2, seed=seed, data_root=path, offline=offline)

        self.X = {'train': data_dict['X']['train'],
                  'train_prior': data_dict['X']['train_prior'],
                  'train_A': data_dict['X']['train_A'],
                  'train_B': data_dict['X']['train_B'],
                  'train_A0': data_dict['X']['train_A0'],
                  'train_B0': data_dict['X']['train_B0'],
                  'train_A1': data_dict['X']['train_A1'],
                  'train_B1': data_dict['X']['train_B1'],
                  'train_0': data_dict['X']['train_0'],
                  'train_1': data_dict['X']['train_1'],

                  'test': data_dict['X']['test'],
                  'test_A': data_dict['X']['test_A'],
                  'test_B': data_dict['X']['test_B'],
                  'test_A0': data_dict['X']['test_A0'],
                  'test_B0': data_dict['X']['test_B0'],
                  'test_A1': data_dict['X']['test_A1'],
                  'test_B1': data_dict['X']['test_B1'],
                  'test_0': data_dict['X']['test_0'],
                  'test_1': data_dict['X']['test_1']}
        self.y = {'train': data_dict['y']['train'],
                  'train_prior': data_dict['y']['train_prior'],
                  'train_A': data_dict['y']['train_A'],
                  'train_B': data_dict['y']['train_B'],
                  'train_A0': data_dict['y']['train_A0'],
                  'train_B0': data_dict['y']['train_B0'],
                  'train_A1': data_dict['y']['train_A1'],
                  'train_B1': data_dict['y']['train_B1'],
                  'train_0': data_dict['y']['train_0'],
                  'train_1': data_dict['y']['train_1'],

                  'test': data_dict['y']['test'],
                  'test_A': data_dict['y']['test_A'],
                  'test_B': data_dict['y']['test_B'],
                  'test_A0': data_dict['y']['test_A0'],
                  'test_B0': data_dict['y']['test_B0'],
                  'test_A1': data_dict['y']['test_A1'],
                  'test_B1': data_dict['y']['test_B1'],
                  'test_0': data_dict['y']['test_0'],
                  'test_1': data_dict['y']['test_1']}

        self.s = {
            'train': data_dict['X']['train'][:, 0],
            'train_prior': None if data_dict['X']['train_prior'] is None else data_dict['X']['train_prior'][:, 0],

            'train_A': data_dict['X']['train_A'][:, 0],
            'train_B': data_dict['X']['train_B'][:, 0],
            'train_A0': data_dict['X']['train_A0'][:, 0],
            'train_B0': data_dict['X']['train_B0'][:, 0],
            'train_A1': data_dict['X']['train_A1'][:, 0],
            'train_B1': data_dict['X']['train_B1'][:, 0],
            'train_0': data_dict['X']['train_0'][:, 0],
            'train_1': data_dict['X']['train_1'][:, 0],

            'test': data_dict['X']['test'][:, 0],
            'test_A': data_dict['X']['test_A'][:, 0],
            'test_B': data_dict['X']['test_B'][:, 0],
            'test_A0': data_dict['X']['test_A0'][:, 0],
            'test_B0': data_dict['X']['test_B0'][:, 0],
            'test_A1': data_dict['X']['test_A1'][:, 0],
            'test_B1': data_dict['X']['test_B1'][:, 0],
            'test_0': data_dict['X']['test_0'][:, 0],
            'test_1': data_dict['X']['test_1'][:, 0],
        }

        if normalize:
            normalization_data = (
                self.X['train_prior']
                if self.X['train_prior'] is not None
                else self.X['train']
            )
            axis = [0] + [
                i + 2
                for i in range(normalization_data.ndim - 2)
            ]
            self.mean = np.mean(
                normalization_data,
                axis=tuple(axis),
                dtype=np.float32,
            )
            self.std = np.std(
                normalization_data,
                axis=tuple(axis),
                dtype=np.float32,
            )

            # if constants, set std to 1
            self.std[self.std == 0.] = 1.

            self.X['train'] = (self.X['train'] - self.mean) / self.std
            if self.X['train_prior'] is not None:
                self.X['train_prior'] = (self.X['train_prior'] - self.mean) / self.std
            self.X['train_A'] = (self.X['train_A'] - self.mean) / self.std
            self.X['train_B'] = (self.X['train_B'] - self.mean) / self.std
            self.X['train_A0'] = (self.X['train_A0'] - self.mean) / self.std
            self.X['train_B0'] = (self.X['train_B0'] - self.mean) / self.std
            self.X['train_A1'] = (self.X['train_A1'] - self.mean) / self.std
            self.X['train_B1'] = (self.X['train_B1'] - self.mean) / self.std
            self.X['train_0'] = (self.X['train_0'] - self.mean) / self.std
            self.X['train_1'] = (self.X['train_1'] - self.mean) / self.std

            self.X['test'] = (self.X['test'] - self.mean) / self.std
            self.X['test_A'] = (self.X['test_A'] - self.mean) / self.std
            self.X['test_B'] = (self.X['test_B'] - self.mean) / self.std
            self.X['test_A0'] = (self.X['test_A0'] - self.mean) / self.std
            self.X['test_B0'] = (self.X['test_B0'] - self.mean) / self.std
            self.X['test_A1'] = (self.X['test_A1'] - self.mean) / self.std
            self.X['test_B1'] = (self.X['test_B1'] - self.mean) / self.std
            self.X['test_0'] = (self.X['test_0'] - self.mean) / self.std
            self.X['test_1'] = (self.X['test_1'] - self.mean) / self.std

        self.data_path = data_path
        self.dataset = dataset


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, s=None):
        self.num_data = len(X)
        self.X = X
        self.y = y[:, None]
        if s is None:
            self.s = None
        else:
            self.s = s if torch.is_tensor(s) else torch.tensor(s).long()

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # Return (x, y, s), preserving access to s after shuffling.
        x = self.X[idx]
        y = self.y[idx]
        s = self.s[idx] if self.s is not None else None
        return x, y, s

    def get_sensitive(self, idx=None):
        if self.s is None:
            return None
        return self.s if idx is None else self.s[idx]
    
