import numpy as np
from collections import Counter
from nutcracker import nutcracker


class SVM:

    def __init__(self, C, gamma):
        self.C = C  # Penalty parameter
        self.gamma = gamma
        self.tolerance = 0.001
        self.max_iter = 40
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = 0

    def kernel_function(self, x, y):
        return np.exp(-np.linalg.norm(x - y)**2 * self.gamma)

    def fit_platt_SMO(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)

        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel_function(X[i], X[j])
        E = np.zeros((n_samples, 2))

        def calcE(i):
            prediction_i = np.sum(self.alpha * y * K[:, i]) + self.b
            error_i = prediction_i - y[i]
            E[i] = [1, error_i]
            return error_i

        def selectJ(i):
            Ei = E[i][1]

            max_delta = 0
            for k in range(n_samples):
                if E[k][0] == 1 and k != i:
                    Ek = E[k][1]
                    delta = abs(Ei - Ek)
                    if delta > max_delta:
                        max_delta = delta
                        max_k = k
                        Ej = Ek

            if max_delta == 0:
                max_k = np.random.randint(0, n_samples - 1)
                while max_k == i:
                    max_k = np.random.randint(0, n_samples - 1)
                Ej = calcE(max_k)
            return max_k, Ej

        def inner(i):
            error_i = calcE(i)
            if y[i] * error_i < -self.tolerance and self.alpha[i] < self.C or y[i] * error_i > self.tolerance and self.alpha[
                    i] > 0:
                j, error_j = selectJ(i)
                alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                if y[i] != y[j]:
                    L = max(0, self.alpha[j] - self.alpha[i])
                    H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                else:
                    L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                    H = min(self.C, self.alpha[i] + self.alpha[j])
                if L == H:
                    return 0

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    return 0

                self.alpha[j] -= (y[j] * (error_i - error_j)) / eta
                self.alpha[j] = np.clip(self.alpha[j], L, H)
                if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                    return 0
                self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                calcE(i)
                calcE(j)

                delta_i = self.alpha[i] - alpha_i_old
                delta_j = self.alpha[j] - alpha_j_old
                b1 = self.b - error_i - y[i] * delta_i * K[i, i] - y[j] * delta_j * K[i, j]
                b2 = self.b - error_j - y[i] * delta_i * K[i, j] - y[j] * delta_j * K[j, j]

                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = (b1 + b2) / 2

                return 1
            else:
                return 0

        cnt = 0
        entire = True
        it = 0
        while it < self.max_iter and (cnt > 0 or entire):
            cnt = 0
            if entire:
                for i in range(n_samples):
                    cnt += inner(i)
                entire = False
            else:
                for i in range(n_samples):
                    if self.C > self.alpha[i] > 0:
                        cnt += inner(i)
            it += 1
            if not cnt:
                entire = True

        mask = (abs(self.alpha) > 0.001)
        self.support_vectors = X[mask]
        self.support_vector_labels = y[mask]
        self.alpha = self.alpha[mask]
        K = K[mask, :][:, mask]
        self.b = (sum(self.support_vector_labels) - sum(self.alpha * self.support_vector_labels * np.sum(K, axis=1))) / len(mask)

    def predict(self, x, mark=False):
        row_sum = np.array([self.kernel_function(x, support_vector) for support_vector in self.support_vectors])
        decision_value = np.sum(self.alpha * self.support_vector_labels * row_sum) + self.b
        if mark:
            return 1 / (1 + np.exp(-decision_value))
        else:
            return 1 if decision_value > 0 else -1


class Meta_MultiClassSVM:

    def __init__(self):
        self.bias = None
        self.types = None

    def fit_meta_bias(self, X_train, y_train):
        self.types = np.unique(y_train)
        num_samples = int(X_train.shape[0] * 0.8)
        num_feature = X_train.shape[1]
        X_train, X_val = X_train[:num_samples], X_train[num_samples:]
        y_train, y_val = y_train[:num_samples], y_train[num_samples:]

        num_classes = len(self.types)
        self.models = np.empty((num_classes, num_classes), dtype=object)
        self.bias = np.empty((num_classes, num_classes), dtype=object)

        lim = [(0.01, 1), (0.1, 0.5)] + [(0, 1)] * num_feature

        meta = nutcracker(2 + num_feature, lim, None)
        meta.best_lim = -1

        cnt = 0
        for i in range(num_classes):
            for j in range(i + 1, num_classes):

                cnt += 1

                mask = (y_train == self.types[i]) | (y_train == self.types[j])
                X_train_1 = X_train[mask]
                y_train_1 = y_train[mask]
                y_train_1 = np.where(y_train_1 == self.types[i], 1, -1)

                mask = (y_val == self.types[i]) | (y_val == self.types[j])
                X_val_1 = X_val[mask]
                y_val_1 = y_val[mask]
                y_val_1 = np.where(y_val_1 == self.types[i], 1, -1)

                def f(x):
                    C, gamma, *y = x
                    y = np.array(y)
                    svm = SVM(C, gamma)
                    svm.fit_platt_SMO(X_train_1 * y, y_train_1)

                    cnt1 = cnt2 = 0
                    for i, j in zip(X_val_1 * y, y_val_1):
                        if svm.predict(i) == j:
                            cnt1 += 1
                        cnt2 += 1
                    return -cnt1 / cnt2

                meta.eval = f
                C, gamma, *y = meta.main_loop()
                print(C, gamma, *y)
                y = np.array(y)
                svm = SVM(C, gamma)
                svm.fit_platt_SMO(np.vstack((X_train_1 * y, X_val_1 * y)), np.hstack((y_train_1, y_val_1)))  ### 注意
                self.models[i, j] = svm
                self.bias[i, j] = y

    def fit(self, X_train, y_train):
        self.types = np.unique(y_train)
        num_feature = X_train.shape[0]
        num_classes = len(self.types)
        self.models = np.empty((num_classes, num_classes), dtype=object)

        cnt = 0
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                cnt += 1

                mask = (y_train == self.types[i]) | (y_train == self.types[j])
                X_train_1 = X_train[mask]
                y_train_1 = y_train[mask]
                y_train_1 = np.where(y_train_1 == self.types[i], 1, -1)

                svm = SVM(1, 1 / num_feature)
                svm.fit_platt_SMO(X_train_1, y_train_1)
                self.models[i, j] = svm

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            votes = Counter()
            num_classes = len(self.models)
            for i in range(num_classes):
                for j in range(i + 1, num_classes):
                    model = self.models[i, j]
                    if self.bias is not None:
                        decision_value = model.predict(x * self.bias[i, j])
                    else:
                        decision_value = model.predict(x)

                    if decision_value == 1:
                        votes[i] += 1
                    else:
                        votes[j] += 1

            predicted_class = votes.most_common(1)[0][0]
            predictions.append(self.types[predicted_class])
        return np.array(predictions)

    def predict_score(self, X_test):
        num_samples = len(X_test)
        num_classes = len(self.models)
        y_score = np.ones((num_samples, num_classes))

        for idx, x in enumerate(X_test):
            for i in range(num_classes):
                for j in range(i + 1, num_classes):
                    model = self.models[i, j]
                    decision_value = model.predict(x, True)
                    y_score[idx, i] *= decision_value
                    y_score[idx, j] *= (1 - decision_value)

            y_score[idx, :] /= y_score[idx, :].sum()

        return y_score
