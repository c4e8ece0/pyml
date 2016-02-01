# 1. Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
import pandas
import numpy as np

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# 2. Оставьте в выборке четыре признака: класс пассажира (Pclass),
#    цену билета (Fare), возраст пассажира (Age) и его пол (Sex).

data.drop(['PassengerId', 'SibSp', 'Name', 'Parch', 'Ticket', 'Cabin', 'Embarked'], inplace=True, axis=1, errors='ignore')

# 3. Обратите внимание, что признак Sex имеет строковые значения.
# 4. Выделите целевую переменную — она записана в столбце Survived.
# 5. В данных есть пропущенные значения — например, для некоторых пассажиров
#    неизвестен их возраст. Такие записи при чтении их в pandas принимают
#    значение nan. Найдите все объекты, у которых есть пропущенные признаки,
#    и удалите их из выборки.
data = data.dropna(axis=0)
data = data.replace({'male': 1, 'female': 2})
Y = data["Survived"]
data.drop(['Survived'], inplace=True, axis=1, errors='ignore')
print data.tail()

# 6. Обучите решающее дерево с параметром random_state=241 и остальными
#    параметрами по умолчанию.
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=241)
clf.fit(data, Y)

# 7. Вычислите важности признаков и найдите два признака с наибольшей
#    важностью. Их названия будут ответами для данной задачи (в качестве
#    ответа укажите названия признаков через запятую или пробел, порядок
#    не важен).
importances = clf.feature_importances_
print importances # Age Fare
