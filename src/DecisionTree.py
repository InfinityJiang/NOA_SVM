import numpy as np


class DecisionTree:

    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        y = y.astype(int)
        self.tree = self.build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self.predict_sample(sample, self.tree) for sample in X])

    def build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return {"label": self.most_common_label(y)}

        best_feature, split_value = self.best_split(X, y)

        if best_feature is None:
            return {"label": self.most_common_label(y)}

        left_idx = X[:, best_feature] < split_value
        right_idx = ~left_idx

        return {
            "feature_idx": best_feature,
            "split_value": split_value,
            "left": self.build_tree(X[left_idx], y[left_idx], depth + 1),
            "right": self.build_tree(X[right_idx], y[right_idx], depth + 1),
        }

    def best_split(self, X, y):
        best_feature = None
        best_split_value = None
        best_gain = -1

        for feature_idx in range(X.shape[1]):
            split_value = np.median(X[:, feature_idx])
            gain = self.information_gain(X[:, feature_idx], y, split_value)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_split_value = split_value

        return best_feature, best_split_value

    def information_gain(self, feature_column, y, split_value):
        parent_entropy = self.entropy(y)
        left_idx = feature_column < split_value
        right_idx = ~left_idx
        if sum(left_idx) == 0 or sum(right_idx) == 0:
            return 0

        n = len(y)
        n_left, n_right = sum(left_idx), sum(right_idx)
        left_entropy = self.entropy(y[left_idx])
        right_entropy = self.entropy(y[right_idx])
        child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        return parent_entropy - child_entropy

    def entropy(self, y):
        y = y.astype(int)
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def predict_sample(self, sample, tree):
        if "label" in tree:
            return tree["label"]
        feature_value = sample[tree["feature_idx"]]
        if feature_value < tree["split_value"]:
            return self.predict_sample(sample, tree["left"])
        else:
            return self.predict_sample(sample, tree["right"])

    def most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
