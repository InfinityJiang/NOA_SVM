import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize, StandardScaler


class evaluate:

    def __init__(self, fit, predict, X, y):
        combined = np.hstack((X, y.reshape(-1, 1)))
        np.random.shuffle(combined)
        X = combined[:, :X.shape[1]]
        y = combined[:, X.shape[1]:].ravel()
        self.fit = fit
        self.predict = predict
        self.X = X
        self.y = y

    def K_fold(self, k):
        scores = []
        fold_size = len(self.X) // k

        for fold in range(k):
            start = fold * fold_size
            end = start + fold_size
            test_x = self.X[start:end]
            test_y = self.y[start:end]
            train_x = np.concatenate([self.X[:start], self.X[end:]])
            train_y = np.concatenate([self.y[:start], self.y[end:]])

            scaler = StandardScaler()
            train_x_scaled = scaler.fit_transform(train_x)
            test_x_scaled = scaler.transform(test_x)

            self.fit(train_x_scaled, train_y)
            predict_y = self.predict(test_x_scaled)

            s = self.compute_metrics(predict_y, test_y)
            scores.append(s)

        return np.mean(scores, axis=0)

    def holdout(self, q):
        split_index = int(len(self.X) * q)
        train_x, test_x = self.X[:split_index], self.X[split_index:]
        train_y, test_y = self.y[:split_index], self.y[split_index:]

        scaler = StandardScaler()
        train_x_scaled = scaler.fit_transform(train_x)
        test_x_scaled = scaler.transform(test_x)

        self.fit(train_x_scaled, train_y)
        predict_y = self.predict(test_x_scaled)

        score = self.compute_metrics(predict_y, test_y)
        return score

    def compute_metrics(self, predictions, y):
        types = np.unique(y)
        num_types = len(types)

        accuracy = np.mean(predictions == y)
        total_samples = len(y)

        precision_list = np.zeros(num_types)
        recall_list = np.zeros(num_types)
        f1_list = np.zeros(num_types)
        weights = np.zeros(num_types)

        for i, c in enumerate(types):
            tp = np.sum((predictions == c) & (y == c))
            fp = np.sum((predictions == c) & (y != c))
            fn = np.sum((predictions != c) & (y == c))

            precision = tp / (tp+fp) if (tp + fp) > 0 else 0
            recall = tp / (tp+fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision+recall) if (precision + recall) > 0 else 0

            precision_list[i] = precision
            recall_list[i] = recall
            f1_list[i] = f1
            weights[i] = (tp+fn) / total_samples

        weighted_precision = np.sum(precision_list * weights)
        weighted_recall = np.sum(recall_list * weights)
        weighted_f1 = np.sum(f1_list * weights)

        return accuracy, weighted_precision, weighted_recall, weighted_f1

    def draw_Confus_mat(self, k):
        scores = []
        fold_size = len(self.X) // k
        all_y_pred = []
        all_y_real = []

        for fold in range(k):
            start = fold * fold_size
            end = start + fold_size
            test_x = self.X[start:end]
            test_y = self.y[start:end]
            train_x = np.concatenate([self.X[:start], self.X[end:]])
            train_y = np.concatenate([self.y[:start], self.y[end:]])

            scaler = StandardScaler()
            train_x_scaled = scaler.fit_transform(train_x)
            test_x_scaled = scaler.transform(test_x)

            self.fit(train_x_scaled, train_y)
            predict_y = self.predict(test_x_scaled)

            s = self.compute_metrics(predict_y, test_y)
            scores.append(s)
            all_y_pred.extend(predict_y)
            all_y_real.extend(test_y)

        cm = confusion_matrix(all_y_pred, all_y_real)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['setosa', 'versicolor', 'virginica'],
            yticklabels=['setosa', 'versicolor', 'virginica']
        )
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title('Confusion Matrix for Iris Dataset')
        plt.show()

    def roc(self, q):
        split_index = int(len(self.X) * q)
        train_x, test_x = self.X[:split_index], self.X[split_index:]
        train_y, test_y = self.y[:split_index], self.y[split_index:]

        scaler = StandardScaler()
        train_x_scaled = scaler.fit_transform(train_x)
        test_x_scaled = scaler.transform(test_x)

        self.fit(train_x_scaled, train_y)
        y_score = self.predict(test_x_scaled)

        n_classes = len(np.unique(self.y))
        if n_classes == 2:
            y_true_binarized = test_y
            y_score_binarized = y_score[:, 1]
        else:
            y_true_binarized = label_binarize(test_y, classes=range(n_classes))
            y_score_binarized = y_score

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            if n_classes == 2:
                fpr[i], tpr[i], _ = roc_curve(y_true_binarized, y_score_binarized)
            else:
                fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_score_binarized[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        mean_auc = auc(all_fpr, mean_tpr)
        plt.plot([0] + list(all_fpr), [0] + list(mean_tpr),
                 color='blue',
                 linestyle='-',
                 label=f'Mean ROC (area = {mean_auc:0.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Average Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
