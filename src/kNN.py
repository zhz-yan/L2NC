import numpy as np
import sklearn
from sklearn.preprocessing import normalize
from scipy.spatial import distance
from scipy.spatial.distance import cdist

def normalized_knn_score(X_train, x_test):
    distances = np.linalg.norm(X_train - x_test, axis=1)
    norms = np.sqrt(np.sum(X_train ** 2, axis=1) + np.linalg.norm(x_test) ** 2 + 1e-8)
    scores = distances / norms
    idx = np.argmin(scores)
    return scores[idx], idx

def cosine_similarity_score(X_train, x_test):
    # 计算训练数据向量的L2范数
    X_train_norm = np.linalg.norm(X_train, axis=1)

    # 计算测试向量的L2范数
    x_test_norm = np.linalg.norm(x_test)

    # 计算余弦相似度 = dot(x_i, x_test) / (||x_i|| * ||x_test||)
    cosine_similarity = np.dot(X_train, x_test) / (X_train_norm * x_test_norm + 1e-8)

    # 将余弦相似度转换为余弦距离
    cosine_distance = 1 - cosine_similarity

    # 找到最小余弦距离（最相似的邻居）
    min_distance = np.min(cosine_distance)
    min_index = np.argmin(cosine_distance)

    return min_distance, min_index

def knn_score_euclidean(X_train, x_test):
    distances = np.linalg.norm(X_train - x_test, axis=1)
    min_distance = np.min(distances)
    min_index = np.argmin(distances)
    return min_distance, min_index

def knn_score_mahalanobis(X_train, x_test, cov_inv):
    distances = cdist(X_train, x_test.reshape(1, -1), metric='mahalanobis', VI=cov_inv).flatten()

    min_distance = np.min(distances)
    min_index = np.argmin(distances)
    return min_distance, min_index

def knn_score_cosine(X_train, x_test):
    X_norm = np.linalg.norm(X_train, axis=1)
    x_test_norm = np.linalg.norm(x_test)
    dot_products = X_train @ x_test

    cosine_sim = dot_products / (X_norm * x_test_norm + 1e-8)
    cosine_distance = 1 - cosine_sim  # the smaller, the closer

    min_distance = np.min(cosine_distance)
    min_index = np.argmin(cosine_distance)
    return min_distance, min_index


def compute_my_scores(X_train, y_train, X_test, y_test, cov_inv):

    tau_list = []
    for i in range(len(X_test)):
        similarity, idx = normalized_knn_score(X_train, X_test[i])
        y_pred = y_train[idx]
        if y_pred == y_test[i]:
            tau_list.append(similarity)

    tau_list = np.array(tau_list)

    return tau_list


def compute_tau_list(X_train, y_train, X_test, y_test):
    tau_list = [
        normalized_knn_score(X_train, x)[0]
        for x, y in zip(X_test, y_test)
        if y_train[normalized_knn_score(X_train, x)[1]] == y
    ]
    return np.array(tau_list)