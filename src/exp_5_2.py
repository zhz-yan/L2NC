import psutil
import time
from pympler import asizeof
from sklearn.metrics import f1_score, \
    roc_auc_score
from features.get_data import get_features
from kNN import *
from evaluation import metric_ood
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
import json
from argparse import Namespace


case_2_config = {
    "P1": {"dataset": "plaid", "unknown": [2, 5]},
    "P2": {"dataset": "plaid", "unknown": [0, 7]},
    "P3": {"dataset": "plaid", "unknown": [0, 5, 9]},
    "P4": {"dataset": "plaid", "unknown": [1, 2, 9]},
    "P5": {"dataset": "plaid", "unknown": [3, 4, 8]},
    "P6": {"dataset": "plaid", "unknown": [6, 7, 9]},
    "P7": {"dataset": "plaid", "unknown": [0, 1, 2, 3]},
    "P8": {"dataset": "plaid", "unknown": [2, 4, 5, 6]},
    "P9": {"dataset": "plaid", "unknown": [5, 6, 7, 8, 10]},
    "P10": {"dataset": "plaid", "unknown": [0, 1, 2, 4, 6, 8]},
    "W1": {"dataset": "whited", "unknown": [2, 3]},
    "W2": {"dataset": "whited", "unknown": [0, 1]},
    "W3": {"dataset": "whited", "unknown": [0, 1, 10]},
    "W4": {"dataset": "whited", "unknown": [1, 3, 7]},
    "W5": {"dataset": "whited", "unknown": [4, 5, 8]},
    "W6": {"dataset": "whited", "unknown": [2, 7, 11]},
    "W7": {"dataset": "whited", "unknown": [3, 4, 6, 8]},
    "W8": {"dataset": "whited", "unknown": [5, 7, 9, 10]},
    "W9": {"dataset": "whited", "unknown": [0, 1, 2, 3]},
    "W10": {"dataset": "whited", "unknown": [3, 4, 6, 7, 9]}
}

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='whited', help="plaid, whited")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./results')
parser.add_argument('--unknown', type=list, default=[0], help='Specify the unknown class.')
parser.add_argument('--pre_save', type=bool, default=True)
parser.add_argument('--feat_num', type=str, default="1")


def case_2(args, exp_idx):
    unknown = args.unknown
    label_count = {"plaid": 11, "cooll": 12, "whited": 12}
    known = list(set(range(label_count[args.dataset])) - set(unknown))
    n_class = len(known)

    data = get_features(args, unknown=unknown, dataset_name=args.dataset)
    X_train, y_train = data['train_data_1'], data['train_y']
    X_test, y_test = data['test_data_1'], data['test_y']
    X_out, y_out = data['out_data_1'], data['out_y']

    rf_clf = RandomForestClassifier(n_estimators=5)
    start_time = time.time()
    rf_clf.fit(X_train, y_train)
    model_size = asizeof.asizeof(rf_clf)
    training_time = time.time() - start_time

    tau_list = compute_my_scores(X_train, y_train, X_test, y_test, n_class)
    percentiles = list(range(90, 100))
    results_all = {}

    for p in percentiles:
        tau = np.percentile(tau_list, p)
        tau_name = f"tau{p}"

        all_X = np.concatenate([X_test, X_out])
        all_y = np.concatenate([y_test, y_out])
        auc_gt = np.concatenate([np.zeros(len(y_test)), np.ones(len(y_out))])

        similarities, preds, preds_rf = [], [], []

        for x in all_X:
            sim, idx = normalized_knn_score(X_train, x)
            pred_rf = rf_clf.predict(x.reshape(1, -1))[0]
            similarities.append(sim)
            preds.append(n_class if sim > tau else pred_rf)
            preds_rf.append(pred_rf)

        preds = np.array(preds)
        similarities = np.array(similarities)
        preds_rf = np.array(preds_rf)

        similarity_k = similarities[:len(y_test)]
        similarity_u = similarities[len(y_test):]

        ood_metrics = metric_ood(similarity_u, similarity_k)['Bas']
        ood_metrics.update({
            'F_macro': f1_score(all_y, preds, average='macro'),
            'F1_unknown': f1_score(all_y, preds, average=None)[-1],
            'AUROC': roc_auc_score(auc_gt, similarities),
            'ACC': np.mean(y_test == preds_rf[:len(y_test)]),
            'tau': tau,
            'training_time': training_time,
            'model_size_MB': model_size / (1024 ** 2)
        })

        results_all[tau_name] = ood_metrics

    return results_all


args = parser.parse_args()


if __name__ == '__main__':

    for case_name, case_info in case_2_config.items():
        dataset = case_info["dataset"]
        unknown = case_info["unknown"]
        for repeat_id in range(10):
            args = Namespace(
                dataset=dataset,
                unknown=unknown
            )
            res = case_2(args, repeat_id)

            json_path = f"results/case_2/{args.dataset}/{str(args.unknown)}/{str(repeat_id)}.json"
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(res, f, indent=4, separators=(",", ": "))
            print(f"Saved results to {json_path}")

        print("All cases finished.")