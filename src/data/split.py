"""Create the fixed client holdout and training folds from project section 5.2."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 4012


def make_splits(client_path, out='data/splits'):
    out = Path(out)
    names = ('train_ids.csv', 'test_ids.csv', 'folds.csv')
    if any((out / name).exists() for name in names):
        raise FileExistsError('Split files already exist. Reuse them; never re-split.')
    clients = pd.read_csv(client_path, usecols=['client_id', 'target'],
                          dtype={'client_id': 'string'}, keep_default_na=False)
    if clients.client_id.str.strip().eq('').any() or clients.client_id.duplicated().any():
        raise ValueError('client_id must be nonempty and unique.')
    clients['target'] = pd.to_numeric(clients.target, errors='raise')
    if set(clients.target.unique()) != {0, 1}:
        raise ValueError('target must contain both binary classes 0 and 1.')
    train, test = train_test_split(clients, test_size=0.2,
                                   stratify=clients.target, random_state=SEED)
    train = train.reset_index(drop=True)
    if train.target.value_counts().min() < 5:
        raise ValueError('Each training class needs at least five clients.')
    folds = train[['client_id']].copy()
    folds['fold'] = -1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, valid_idx) in enumerate(cv.split(train, train.target)):
        folds.loc[valid_idx, 'fold'] = fold
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(names, (train[['client_id']], test[['client_id']], folds)):
        frame.to_csv(out / name, index=False, mode='x')
    print(f'Saved {len(train):,} train clients, {len(test):,} test clients and five training folds to {out}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clients', default='data/raw/client_train.csv')
    parser.add_argument('--out', default='data/splits')
    args = parser.parse_args()
    make_splits(args.clients, args.out)
