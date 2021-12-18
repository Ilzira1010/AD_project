# AD_project

Всем привет! В своем проекте мы хотели рассмотреть набор данных,который посвящен стоимости лечения разных пациентов. Как все мы знаем стоимость лечения зависит от многих факторов: диагноза, типа клиники, города проживания, возраста и так далее. Данных о диагнозе пациентов у нас нет. Но у нас есть и другая информация, которая может помочь нам сделать вывод о здоровье пациентов и провести регрессионный анализ. Мы желаем всем Вам не болеть. Давайте рассмотрим наши данные.

image.png

[ ]
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as pl
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

Чтобы изменить содержимое ячейки, дважды нажмите на нее (или выберите "Ввод")

[ ]
data = pd.read_csv('/content/insurance.csv', parse_dates = True)
new_data = data
new_data.head()


[ ]
new_data.isnull().sum()

age         0
sex         0
bmi         0
children    0
smoker      0
region      0
charges     0
dtype: int64
Нам крупно повезло,потому что в датасете нет "Nan"))))) Давайте посмотрим на наши данные, чтобы что-то понять. Поскольку нас в первую очередь интересует сумма затрат, посмотрим, какие данные больше коррелируют с расходами. Для начала закодируем категориальные признаки.

[ ]
from sklearn.preprocessing import LabelEncoder
#пол
le = LabelEncoder()
le.fit(new_data.sex.drop_duplicates()) 
new_data.sex = le.transform(new_data.sex)
# курящий или нет
le.fit(new_data.smoker.drop_duplicates()) 
new_data.smoker = le.transform(new_data.smoker)
#регион
le.fit(new_data.region.drop_duplicates()) 
new_data.region = le.transform(new_data.region)
Несколько слов о кодировке «region». Как правило, категориальные переменные имеют своиство неустоичивости и их лучше всего кодировать с помощью OneHotEncoder и так далее. Но в этом случае ничего не изменится, потому что нет особого порядка, в котором были бы перечислены регионы.

[ ]
new_data.corr()['charges'].sort_values()

region     -0.006208
sex         0.057292
children    0.067998
bmi         0.198341
age         0.299008
smoker      0.787251
charges     1.000000
Name: charges, dtype: float64
[ ]
f, ax = pl.subplots(figsize=(10, 8))
corr = new_data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),
            square=True, ax=ax)


Как мы можем заметить сильная корреляция наблюдается только с фактом курения больного. Мы думали,что высочайшая корреляция будет с BMI(Индекс массы тела). Давайте подробнее рассмотрим курение.

CoarseAdventurousIbis-max-1mb.gif

[ ]
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
output_notebook()
import scipy.special
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import row, column

[ ]
p = figure(title="Распределение расходов",tools="save",background_fill_color="#E8DDCB")
hist, edges = np.histogram(new_data.charges)
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],fill_color="#036564", line_color="#033649")
p.xaxis.axis_label = 'x'
p.yaxis.axis_label = 'Pr(x)'
#show(gridplot(p,ncols = 2, plot_width=400, plot_height=400, toolbar_location=None))
show(row(p))
#show(p)


[ ]
f= pl.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.smoker == 1)]["charges"],color='c',ax=ax)
ax.set_title('Распределение сборов для курильщиков')

ax=f.add_subplot(122)
sns.distplot(data[(data.smoker == 0)]['charges'],color='b',ax=ax)
ax.set_title('Распределение платы для некурящих')


Как мы видим курильщики тратят больше денег на лечение. Но мы все же считаем,что некурящих пациентов больше. Давайте это проверим

[ ]
sns.catplot(x="smoker", kind="count",hue = 'sex', palette="pink", data=data)


image.png

Просим обратить внимание, что женщины кодируются символом "1", а мужчины - "0". При этом некурящих людей и правда больше. Также мы можем заметить, что курящих мужчин больше, чем курящих женщин. Можно предположить, что общая стоимость лечения у мужчин будет больше, чем у женщин, учитывая влияние курения. И еще несколько полезных визуализаций

[ ]
sns.catplot(x="sex", y="charges", hue="smoker",
            kind="violin", data=data, palette = 'magma')


[ ]
pl.figure(figsize=(12,5))
pl.title("Ящик с усами для женщин")
sns.boxplot(y="smoker", x="charges", data =  data[(data.sex == 1)] , orient="h", palette = 'magma')


[ ]
pl.figure(figsize=(12,5))
pl.title("Ящик с усами для мужчин")
sns.boxplot(y="smoker", x="charges", data =  data[(data.sex == 0)] , orient="h", palette = 'rainbow')


Теперь давайте обратим внимание на возраст пациентов. Во-первых, давайте посмотрим, как возраст влияет на стоимость лечения, а также посмотрим на пациентов, какого возраста больше в нашем наборе данных.

[ ]
pl.figure(figsize=(12,5))
pl.title("Распределение по возрасту")
ax = sns.distplot(data["age"], color = 'g')


В нашем наборе данных есть пациенты младше 20 лет. Мы взяли за минимальный возраст 18 лет,тк это возраст совершеннолетия и вроде как курение не запрещено..... Максимальный возраст - 64 года. Нас интересует, есть ли курильщики среди пациентов 18 лет.

[ ]
sns.catplot(x="smoker", kind="count",hue = 'sex', palette="rainbow", data=data[(data.age == 18)])
pl.title("Количество курящих и некурящих (18 лет)")
Мдааа.... А мы надеялись, что результат будет другим. Давайте выясним влияет ли курение на стоимость лечения в этом возрасте?

image.png

[ ]
pl.figure(figsize=(12,5))
pl.title("Ящик с усами для курильщиков в возрасте 18 лет")
sns.boxplot(y="smoker", x="charges", data = data[(data.age == 18)] , orient="h", palette = 'pink')


Как мы видим,даже в возрасте 18 лет курильщики тратят на лечение гораздо больше,чем некурящие.Среди некурящих мы наблюдаем некоторые аномалии. Мы предполагаем,что это эти аномалии - это несчастные случаи и серьезные заболевания.Теперь давайте посмотрим, как стоимость лечения зависит от возраста курильщиков и некурящих пациентов.

[ ]
g = sns.jointplot(x="age", y="charges", data = data[(data.smoker == 0)],kind="kde", color="m")
g.plot_joint(pl.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$")
ax.set_title('Распределение расходов и возраста для некурящих')


[ ]
g = sns.jointplot(x="age", y="charges", data = data[(data.smoker == 1)],kind="kde", color="c")
g.plot_joint(pl.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$")
ax.set_title('Распределение расходов и возраста для курильщиков')


[ ]
#некурящие
p = figure(plot_width=500, plot_height=450)
p.circle(x=data[(data.smoker == 0)].age,y=data[(data.smoker == 0)].charges, size=7, line_color="navy", fill_color="pink", fill_alpha=0.9)

show(p)


[ ]
#курящие
p = figure(plot_width=500, plot_height=450)
p.circle(x=data[(data.smoker == 1)].age,y=data[(data.smoker == 1)].charges, size=7, line_color="navy", fill_color="red", fill_alpha=0.9)
show(p)
[ ]
sns.lmplot(x="age", y="charges", hue="smoker", data=data, palette = 'inferno_r', size = 7)
ax.set_title('Курящие и некурящие')
У некурящих стоимость лечения увеличивается с возрастом.Оно и понятно! Так что берегите свое здоровье, друзья! У курящих людей мы не видим такой зависимости. Мы думаем, что дело не только в курении, но и в особенностях набора данных. О таком сильном влиянии курения на стоимость лечения логичнее было бы судить, имея набор данных с большим количеством записей и знаков. Но мы работаем с тем, что у нас есть! Давайте обратим внимание на ИМТ.

[ ]
pl.figure(figsize=(12,5))
pl.title("Распределение ИМТ")
ax = sns.distplot(data["bmi"], color = 'm')


Мы видим очень красивый график. В среднем ИМТ у пациентов из выборки 30. Давайте загуглим этот показатель))))

image.png

При значении, равном 30, начинается ожирение. Давайте посмотрим на распределение затрат у пациентов с ИМТ более 30 и менее 30.

[ ]
pl.figure(figsize=(12,5))
pl.title("Распределение расходов для пациентов с ИМТ более 30")
ax = sns.distplot(data[(data.bmi >= 30)]['charges'], color = 'm')



[ ]
pl.figure(figsize=(12,5))
pl.title("Распределение расходов для пациентов с ИМТ менее 30")
ax = sns.distplot(data[(data.bmi < 30)]['charges'], color = 'b')


Пациенты с ИМТ выше 30 тратят больше на лечение!

[ ]
g = sns.jointplot(x="bmi", y="charges", data = data,kind="kde", color="r")
g.plot_joint(pl.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$")
ax.set_title('Распределение ИМТ и расходов')


[ ]
pl.figure(figsize=(10,6))
ax = sns.scatterplot(x='bmi',y='charges',data=data,palette='magma',hue='smoker')
ax.set_title('Точечный график расходов и ИМТ')

sns.lmplot(x="bmi", y="charges", hue="smoker", data=data, palette = 'magma', size = 8)


Давайте обратим внимание на детей. Во-первых, давайте посмотрим, сколько детей у наших пациентов.

[ ]
sns.catplot(x="children", kind="count", palette="ch:.25", data=data, size = 6)


У большинства пациентов нет детей. Интересно, курят ли люди, у которых есть дети?

[ ]
sns.catplot(x="smoker", kind="count", palette="rainbow",hue = "sex",
            data=data[(data.children > 0)], size = 6)
ax.set_title('Курильщики и некурящие, у которых есть дети')


Мы рады, что некурящих родителей гораздо больше!

Теперь мы собираемся спрогнозировать стоимость лечения. Давайте начнем с обычной линейной регрессии.

[ ]
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
[ ]
x = data.drop(['charges'], axis = 1)
y = data.charges

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)
lr = LinearRegression().fit(x_train,y_train)

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

print(lr.score(x_test,y_test))

0.7962732059725786
Неплохо.... Теперь давайте добавим полиномиальные знаки. И посмотрите на результат.

[ ]
X = data.drop(['charges','region'], axis = 1)
Y = data.charges



quad = PolynomialFeatures (degree = 2)
x_quad = quad.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(x_quad,Y, random_state = 0)

plr = LinearRegression().fit(X_train,Y_train)

Y_train_pred = plr.predict(X_train)
Y_test_pred = plr.predict(X_test)

print(plr.score(X_test,Y_test))

0.8849197344147228
Хорошо!Наша модель хорошо прогнозирует стоимость лечения пациентов. И, наконец, попробуем RandomForestRegressor.

[ ]
forest = RandomForestRegressor(n_estimators = 100,
                              criterion = 'mse',
                              random_state = 1,
                              n_jobs = -1)
forest.fit(x_train,y_train)
forest_train_pred = forest.predict(x_train)
forest_test_pred = forest.predict(x_test)

print('MSE train data: %.3f, MSE test data: %.3f' % (
mean_squared_error(y_train,forest_train_pred),
mean_squared_error(y_test,forest_test_pred)))
print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,forest_train_pred),
r2_score(y_test,forest_test_pred)))

MSE train data: 3729086.094, MSE test data: 19933823.142
R2 train data: 0.974, R2 test data: 0.873
[ ]
pl.figure(figsize=(10,6))

pl.scatter(forest_train_pred,forest_train_pred - y_train,
          c = 'black', marker = 'o', s = 35, alpha = 0.5,
          label = 'Train data')
pl.scatter(forest_test_pred,forest_test_pred - y_test,
          c = 'c', marker = 'o', s = 35, alpha = 0.7,
          label = 'Test data')
pl.xlabel('Predicted values')
pl.ylabel('Tailings')
pl.legend(loc = 'upper left')
pl.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'red')
pl.show()


Хороший результат. Но мы видим заметную переподготовку алгоритма на обучающих данных.

Всем спасибо за внимание!

image.png
