import psutil
import time
from pympler import asizeof
from sklearn.metrics import f1_score, \
    precision_recall_fscore_support, roc_auc_score
from features.get_data import get_features
from kNN import *
from evaluation import metric_ood
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
import json
from utils import get_known_labels

parser = argparse.ArgumentParser("Training")

def case_1(args):
    unknown = args.unknown
    known = get_known_labels(args.dataset, unknown)
    n_class = len(np.unique(known))

    data = get_features(args, unknown=unknown, dataset_name=args.dataset)
    X_train, y_train = data['train_data_1'], data['train_y']
    X_test, y_test = data['test_data_1'], data['test_y']
    X_out, y_out = data['out_data_1'], data['out_y']

    rf_clf = RandomForestClassifier(n_estimators=5)
    start_time = time.time()
    rf_clf.fit(X_train, y_train)
    model_size = asizeof.asizeof(rf_clf)
    training_time = time.time() - start_time

    tau_list = compute_tau_list(X_train, y_train, X_test, y_test)
    percentiles = range(90, 100)
    results_all = {}

    for p in percentiles:
        tau = np.percentile(tau_list, p)
        pred_all, sim_all = [], []

        for X, is_unknown in [(X_test, False), (X_out, True)]:
            for x in X:
                sim, idx = normalized_knn_score(X_train, x)
                y_pred = rf_clf.predict(x.reshape(1, -1))[0]
                is_open = sim > tau
                pred_all.append(n_class if is_open else y_pred)
                sim_all.append(sim)

        pred_all = np.array(pred_all)
        sim_all = np.array(sim_all)
        label_gt = np.concatenate((y_test, y_out))
        auc_gt = np.concatenate((np.zeros(len(y_test)), np.ones(len(y_out))))

        ood_metrics = metric_ood(
            x1=sim_all[len(y_test):],
            x2=sim_all[:len(y_test)]
        )['Bas']

        ood_metrics.update({
            'F_macro': f1_score(label_gt, pred_all, average='macro'),
            'F1_unknown': f1_score(label_gt, pred_all, average=None)[-1],
            'precision': precision_recall_fscore_support(label_gt, pred_all, average='macro')[0],
            'recall': precision_recall_fscore_support(label_gt, pred_all, average='macro')[1],
            'AUROC': roc_auc_score(auc_gt, sim_all),
            'ACC': np.mean(y_test == rf_clf.predict(X_test)),
            'tau': tau,
            'training_time': training_time,
            'model_size_MB': model_size / (1024 ** 2)
        })
        results_all[f"tau{p}"] = ood_metrics

    return results_all

parser.add_argument('--dataset', type=str, default='whited', help="plaid, whited")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./results')
parser.add_argument('--unknown', type=list, default=[0], help='Specify the unknown class.')
parser.add_argument('--pre_save', type=bool, default=True)
parser.add_argument('--feat_set', type=str, default="currentF")
args = parser.parse_args()


if __name__ == '__main__':

    # Set unknown class to [12] when using the 'whited' dataset.
    args.dataset = "plaid"
    for unknown_class in range(11):
        args.unknown = [unknown_class]
        for repeat_id in range(1):
            res = case_1(args, repeat_id)

            json_path = f"results/case_1/{args.dataset}/{str(args.unknown[0])}/{str(repeat_id)}.json"
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(res, f, indent=4, separators=(",", ": "))
            print(f"Saved results to {json_path}")
