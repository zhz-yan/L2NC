from features.get_features import create_features
from sklearn.preprocessing import LabelEncoder
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
# from gao_features import gao_features

def split_open_set(data, labels, known_classes, test_size=0.2):
    # Convert known_classes to a set for faster membership testing
    known_classes_set = set(known_classes)

    # Identify known and unknown indices based on labels
    known_indices = [i for i, label in enumerate(labels) if label in known_classes_set]
    unknown_indices = [i for i, label in enumerate(labels) if label not in known_classes_set]

    # Split known data
    known_data = data[known_indices]
    known_labels = labels[known_indices]

    # Split unknown data
    unknown_data = data[unknown_indices]
    unknown_labels = labels[unknown_indices]

    # Split the known data into training and testing sets
    train_X, test_known_X, train_y, test_known_y = train_test_split(known_data, known_labels, test_size=test_size, stratify=known_labels)

    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2)

    # Normalization harmonics
    scaler = MinMaxScaler()
    # scaler = StandardScaler()

    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_known_X = scaler.transform(test_known_X)
    unknown_data = scaler.transform(unknown_data)

    data = {"train_data_1": train_X,
            "train_y": train_y,
            "val_data": val_X,
            "val_y": val_y,
            "test_data_1": test_known_X,
            "test_y": test_known_y,
            "out_data_1": unknown_data,
            "out_y": unknown_labels
    }

    return data


def get_features(args, unknown, dataset_name: str = 'plaid'):

    # get raw data
    if dataset_name == 'plaid':
        current = np.load('data/plaid/current.npy')
        voltage = np.load('data/plaid/voltage.npy')
        labels = np.load('data/plaid/labels.npy')
        known = list(set(list(range(0, 11))) - set(unknown))
        fs = 30e3
        f0 = 60

    elif dataset_name == 'whited':
        current = np.load('data/whited/current.npy')
        voltage = np.load('data/whited/voltage.npy')
        labels = np.load('data/whited/labels.npy')
        known = list(set(list(range(0, 12))) - set(unknown))
        fs = 44.1e3
        f0 = 50         # in most cases

    # clf_feat = gao_features(args, voltage, current, fs, f0)

    # get features
    clf_feat = create_features(voltage, current, fs)

    # split the data
    data = split_open_set(clf_feat, labels, known)

    print(f"train set labels:{np.unique(data['train_y'])}")
    # print(f"validation set labels:{np.unique(data['val_y'])}")
    print(f"test set labels:{np.unique(data['test_y'])}")
    print(f"Out set labels:{np.unique(data['out_y'])}")
    print(f"numbers of training set: {len(data['train_y'])}")
    print(f"numbers of test set: {len(data['test_y']) + len(data['out_y'])}")

    le = LabelEncoder()
    le.fit(data['train_y'])

    data['train_y'] = le.transform(data['train_y'])
    data['val_y'] = le.transform(data['val_y'])
    data['test_y'] = le.transform(data['test_y'])
    data['out_y'][:] = len(np.unique(data['train_y']))

    return data


