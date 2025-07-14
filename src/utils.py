
def get_known_labels(dataset, unknown):
    label_count = {"plaid": 11, "whited": 12}
    return list(set(range(label_count[dataset])) - set(unknown))