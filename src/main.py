from sklearn.preprocessing import StandardScaler
from svm import Meta_MultiClassSVM
from DecisionTree import DecisionTree
from evaluate import evaluate

# 使用Iris测试集
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

# # 使用breast cancer数据集
# from sklearn.datasets import load_breast_cancer

# breast_cancer = load_breast_cancer()
# X, y = breast_cancer.data, breast_cancer.target

# k折交叉验证-NOASVM
model = Meta_MultiClassSVM()
evaluater = evaluate(model.fit_meta_bias, model.predict, X, y)
result = evaluater.K_fold(10)

# # k折交叉验证-普通SVM
# model = Meta_MultiClassSVM()
# evaluater = evaluate(model.fit, model.predict, X, y)
# result = evaluater.K_fold(10)

# # k折交叉验证-普通决策树
# model = DecisionTree()
# evaluater = evaluate(model.fit, model.predict, X, y)
# result = evaluater.K_fold(10)

# # 留出法
# model = Meta_MultiClassSVM()
# evaluater = evaluate(model.fit_meta_bias, model.predict, X, y)
# result = evaluater.hold_out(0.8)

# # 绘制混淆矩阵 (基于k折交叉验证)
# model = Meta_MultiClassSVM()
# evaluater = evaluate(model.fit, model.predict, X, y)
# result = evaluater.draw_Confus_mat(10)

# # 绘制ROC曲线 (基于留出法，支持多分类)
# model = Meta_MultiClassSVM()
# evaluater = evaluate(model.fit, model.predict_score, X, y)
# result = evaluater.roc(0.7)

print(result)
