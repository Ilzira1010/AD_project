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

we use one hot encoding by using `get_dummies()`

![one hot encoding](assets/one-hot-encoding.JPG)


We try to then find the correlation between features.

![eda](assets/eda.JPG)

Using a heat map to explore the trends.

![heatmap](assets/corr-heatmap.JPG)

From this we can see the following observations:

1. Strong correlation between charges and smoker_yes.
2. Weak correlation between charges and age.
3. Weak correlation between charges and bmi.
4. Weak correlation between bmi and region_southeast.
Since the values for the weak correlations are less than 0.5, we can term them as insignificant and drop them.

Exploring the trend between charges and smoker_yes.
Finding the range of the treatment charges of patients using graphs.

![range of charges](assets/charges_range.JPG)

From the graph, We can see the minimum charges are around 1122 for a high number of patients and maximum of 63770.

### 4. Model Building

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
