![GitHub top language](https://img.shields.io/github/languages/top/Thomas-George-T/Regression-on-Personal-Health-Data)
![GitHub last commit](https://img.shields.io/github/last-commit/Thomas-George-T/Regression-on-Personal-Health-Data?style=flat)
![GitHub License](https://img.shields.io/github/license/Thomas-George-T/Regression-on-Personal-Health-Data?style=flat)
![ViewCount](https://views.whatilearened.today/views/github/Thomas-George-T/Regression-on-Personal-Health-Data.svg?cache=remove)

# Regression on Personal Health Data

## Aim

Predicting the cost of treatment and insurance using regression by leveraging personal health data.

## Motivation
Hi all, This is my first notebook. I am trying to perform Exploratory Data Analysis (EDA) and linear regression on personal health data. Any feedback and constructive criticism is appreciated. The personal heath data is hosted on Kaggle. Link: https://www.kaggle.com/mirichoi0218/insurance

## Table of contents
1. [Components](https://github.com/Thomas-George-T/Linear-Regression-on-Personal-Health-Data#Components)
2. [Model Implementation](https://github.com/Thomas-George-T/Linear-Regression-on-Personal-Health-Data#model-implementation)
   1. [Import Data](https://github.com/Thomas-George-T/Linear-Regression-on-Personal-Health-Data#1-import-data)
   2. [Data Preprocessing](https://github.com/Thomas-George-T/Linear-Regression-on-Personal-Health-Data#2-data-preprocessing)
   3. [Exploratory Data Analysis (EDA)](https://github.com/Thomas-George-T/Linear-Regression-on-Personal-Health-Data#3-exploratory-data-analysis-eda)
   4. [Model Building](https://github.com/Thomas-George-T/Linear-Regression-on-Personal-Health-Data#4-model-building)
   5. [Model Evaluation](https://github.com/Thomas-George-T/Linear-Regression-on-Personal-Health-Data#5-model-evaluation)
3. [License](https://github.com/Thomas-George-T/Linear-Regression-on-Personal-Health-Data#License)
  
## Components
- [Kaggle Dataset](https://www.kaggle.com/mirichoi0218/insurance)
- Jupyter notebook
- Python: numpy, pandas, matplotlib packages

## Model Implementation

### 1. Import Data

Как только мы импортируем данные с помощью read_csv, используйте head() для выборки данных. Мы пытаемся идентифицировать числовые и категориальные данные.

![0](assers/0.jpg)

### 2. Data Preprocessing

Нам крупно повезло,потому что в датасете нет "Nan"))))) Давайте посмотрим на наши данные, чтобы что-то понять. Поскольку нас в первую очередь интересует сумма затрат, посмотрим, какие данные больше коррелируют с расходами. Для начала закодируем категориальные признаки.

```python
new_data.isnull().sum()
```

![16](assers/16.jpg)

```python
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
```
Несколько слов о кодировке «region». Как правило, категориальные переменные имеют своиство неустоичивости и их лучше всего кодировать с помощью OneHotEncoder и так далее. Но в этом случае ничего не изменится, потому что нет особого порядка, в котором были бы перечислены регионы.


### 3. Exploratory Data Analysis (EDA)
```python
new_data.corr()['charges'].sort_values()
```
![1](assers/1.jpg)

Как мы можем заметить сильная корреляция наблюдается только с фактом курения больного. Мы думали,что высочайшая корреляция будет с BMI(Индекс массы тела). Давайте подробнее рассмотрим курение.

```python
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
output_notebook()
import scipy.special
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import row, column
```
![2](assers/2.jpg)
![3](assers/3.jpg)

Как мы видим курильщики тратят больше денег на лечение. Но мы все же считаем,что некурящих пациентов больше. Давайте это проверим

![4](assers/4.jpg)


Просим обратить внимание, что женщины кодируются символом "1", а мужчины - "0". При этом некурящих людей и правда больше. Также мы можем заметить, что курящих мужчин больше, чем курящих женщин. Можно предположить, что общая стоимость лечения у мужчин будет больше, чем у женщин, учитывая влияние курения. И еще несколько полезных визуализаций

![5](assers/5.jpg)
![6](assers/6.jpg)

Теперь давайте обратим внимание на возраст пациентов. Во-первых, давайте посмотрим, как возраст влияет на стоимость лечения, а также посмотрим на пациентов, какого возраста больше в нашем наборе данных.

![7](assers/7.jpg)

В нашем наборе данных есть пациенты младше 20 лет. Мы взяли за минимальный возраст 18 лет,тк это возраст совершеннолетия и вроде как курение не запрещено..... Максимальный возраст - 64 года. Нас интересует, есть ли курильщики среди пациентов 18 лет.
![8](assers/8.jpg)

Мдааа.... А мы надеялись, что результат будет другим. Давайте выясним влияет ли курение на стоимость лечения в этом возрасте?

![9](assers/9.jpg)

Как мы видим,даже в возрасте 18 лет курильщики тратят на лечение гораздо больше,чем некурящие.Среди некурящих мы наблюдаем некоторые аномалии. Мы предполагаем,что это эти аномалии - это несчастные случаи и серьезные заболевания.Теперь давайте посмотрим, как стоимость лечения зависит от возраста курильщиков и некурящих пациентов.
![10](assers/10.jpg)
![11](assers/11.jpg)
![12](assers/12.jpg)
![13](assers/13.jpg)
![14](assers/14.jpg)

У некурящих стоимость лечения увеличивается с возрастом.Оно и понятно! Так что берегите свое здоровье, друзья! У курящих людей мы не видим такой зависимости. Мы думаем, что дело не только в курении, но и в особенностях набора данных. О таком сильном влиянии курения на стоимость лечения логичнее было бы судить, имея набор данных с большим количеством записей и знаков. Но мы работаем с тем, что у нас есть! Давайте обратим внимание на ИМТ.
![15](assers/15.jpg)

Мы видим очень красивый график. В среднем ИМТ у пациентов из выборки 30. Давайте загуглим этот показатель))))
![17](assers/17.jpg)

При значении, равном 30, начинается ожирение. Давайте посмотрим на распределение затрат у пациентов с ИМТ более 30 и менее 30.
![18](assers/18.jpg)

Пациенты с ИМТ выше 30 тратят больше на лечение!
![19](assers/19.jpg)
![20](assers/20.jpg)

Давайте обратим внимание на детей. Во-первых, давайте посмотрим, сколько детей у наших пациентов.
![21](assers/21.jpg)

У большинства пациентов нет детей. Интересно, курят ли люди, у которых есть дети?
![22](assers/22.jpg)

Мы рады, что некурящих родителей гораздо больше!

Теперь мы собираемся спрогнозировать стоимость лечения. Давайте начнем с обычной линейной регрессии.

### 4. Model Building
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
x = data.drop(['charges'], axis = 1)
y = data.charges

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)
lr = LinearRegression().fit(x_train,y_train)

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

print(lr.score(x_test,y_test))
```
```python
0.7962732059725786
```


![Model building](assets/model-building.JPG)

We then begin to predict the values of the patient charges using the other features. We build a linear regression model after importing the package `sklearn.linear_model`. We split the data set into training and test set. We use 30% of the dataset for testing using `test_size=0.3` 
We take the predictor variable without the charges column and the target variable as charges.
We proceed to fit the linear regression model for the test and training set using `fit()`. This part is called **Model fitting**. We check the prediction score of both and training and test set using `score()`. It comes out to be 79%, which is pretty decent I would say.

### 5. Model Evaluation

To evaluate our linear regression, we use R<sup>2</sup> and mean squared error.

![mode evaluation](assets/model-evaluation.JPG)

On evaluating our model, it showed <bold>accuracy of 80%</bold> on the test data. 

From the figure, Our evaluation metrics of R<sup>2</sup> and mean squared error of both training and test data are closely matching. This is enough to conclude our model is appropriate to predict patient charges based on their personal health data.

## License
This project is under the MIT License - see [License](LICENSE.md) for more details
