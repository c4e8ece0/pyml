﻿# Пример загрузки данных в Pandas:
# src https://www.kaggle.com/c/titanic/data

import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# Данные будут загружены в виде DataFrame, с помощью которого можно
# удобно работать с ними. В данном случае параметр
# index_col='PassengerId’
# означает, что колонка PassengerId задает нумерацию строк данного
# датафрейма

# Для того, чтобы посмотреть что представляют из себя данные, можно
# воспользоваться несколькими способами:

# 1. более привычным с точки зрения Python (если индекс указывается
# только один, то производится выбор строк):
data[:10]

# 2. или же воспользоваться методом датафрейма:
data.head()

# Один из способов доступа к столбцам датафрейма — использовать
# квадратные скобки и название столбца:
data['Pclass']
# print data['Pclass'].value_counts()

# 1. Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите
# два числа через пробел.
print "1 --------------------------------\n"
print"Sex:\n"
print data['Sex'].value_counts() # 577 314

# 2. Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен).
print "\n\n2 --------------------------------\n"
total = data['Survived'].size*1.0
live = data['Survived'].sum()*1.0
print("Survive: %d\n", live)
print("Total: %d\n", total)
print "Survived %:"
print live/total*100.0

# 3. Какую долю пассажиры первого класса составляли среди всех пассажиров?
# Ответ # приведите в процентах (число в интервале от 0 до 100, знак процента
# не нужен).
print "\n\n3 --------------------------------\n"
print data['Pclass'].value_counts()
fc = data['Pclass'].value_counts()[1]*1.0
print fc/total*100

# 4. Какого возраста были пассажиры? Посчитайте среднее и медиану возраста
# пассажиров. В качестве ответа приведите два числа через пробел.
print "\n\n4 --------------------------------\n"
print("Mean: %.2f", data["Age"].mean())
print("Median: %.2f", data["Age"].median())

# 5. Коррелируют ли число братьев/сестер с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
print "\n\n5 --------------------------------\n"
print data.corr("pearson")

# 6. Какое самое популярное женское имя на корабле? Извлеките из полного
# имени пассажира (колонка Name) его личное имя (First Name). Это
# задание — типичный пример того, с чем сталкивается специалист по анализу
# данных. Данные очень разнородные и шумные, но из них требуется извлечь
# необходимую информацию. Попробуйте вручную разобрать несколько значений
# столбца Name и выработать правило для извлечения имен, а также разделения
# их на женские и мужские.
print "\n\n6 --------------------------------\n"

from collections import defaultdict

def leaders(xs, top=100):
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    return sorted(counts.items(), reverse=True, key=lambda tup: tup[1])[:top]

cnt = []
for index, row in data.iterrows():
    for name in row["Name"].split(" "):
        cnt[len(cnt):] = [name]

print leaders(cnt)
