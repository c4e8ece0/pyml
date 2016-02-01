# Одна из особенностей решающих деревьев заключается в том, что они позволяют
# получать важности всех используемых признаков. Важность признака можно
# оценить на основе того, как сильно улучшился критерий качества благодаря
# использованию этого признака в вершинах дерева.

# РЕАЛИЗАЦИЯ В SCIKIT-LEARN
# В библиотеке scikit-learn решающие деревья реализованы в классах
# sklearn.tree.DecisionTreeСlassifier (для классификации)
# и sklearn.tree.DecisionTreeRegressor (для регрессии). Обучение модели
# производится с помощью функции fit.

import numpy as np
from sklearn.tree import DecisionTreeClassifier

X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
clf = DecisionTreeClassifier()
clf.fit(X, y)
importances = clf.feature_importances_

# Переменная importances будет содержать массив "важностей" признаков. Индекс
# в этом массиве соответствует индексу признака в данных.
print importances

# Стоит обратить внимание, что данные могут содержать пропуски. Pandas хранит
# такие значения как nan (not a number). Для того, чтобы проверить, является
# ли число nan'ом, можно воспользоваться функцией np.isnan.
print np.isnan(X)
