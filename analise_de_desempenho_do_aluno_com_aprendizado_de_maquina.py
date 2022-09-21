
# para algumas operações básicas
import numpy as np
import pandas as pd

# para visualizações
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import dabl

data = pd.read_csv('StudentsPerformance.csv')

# obtendo a forma dos dados
print(data.shape)

data.head(10)

data.describe()

data.columns

data.select_dtypes('object').nunique()

no_of_columns = data.shape[0]
percentage_of_missing_data = data.isnull().sum()/no_of_columns
print(percentage_of_missing_data)

data[['lunch','gender','math score','writing score',
      'reading score']].groupby(['lunch','gender']).agg('median')

data[['test preparation course',
      'gender',
      'math score',
      'writing score',
      'reading score']].groupby(['test preparation course','gender']).agg('median')


import warnings
warnings.filterwarnings('ignore')

data['total_score'] = data['math score'] + data['reading score'] + data['writing score']


# importing math library to use ceil
from math import * 
import warnings
warnings.filterwarnings('ignore')

data['percentage'] = data['total_score']/3

for i in range(0, 1000):
    data['percentage'][i] = ceil(data['percentage'][i])

data

data.columns

def getgrade(percentage):
  if(percentage >= 90):
    return 'O'
  if(percentage >= 80):
    return 'A'
  if(percentage >= 70):
    return 'B'
  if(percentage >= 60):
    return 'C'
  if(percentage >= 40):
    return 'D'
  else :
    return 'E'

data['grades'] = data.apply(lambda x: getgrade(x['percentage']), axis = 1 )
data['grades'].value_counts()

data.columns

data

from sklearn.preprocessing import LabelEncoder

# criando um codificador
le_cod = LabelEncoder()

# codificação de etiquetas para curso de preparação para testes
data['test preparation course'] = le_cod.fit_transform(data['test preparation course'])

# codificação de rótulos para almoço
data['lunch'] = le_cod.fit_transform(data['lunch'])

# codificação de rótulo para raça/etnia
# temos que mapear valores para cada uma das categorias
data['race/ethnicity'] = data['race/ethnicity'].replace('group A', 1)
data['race/ethnicity'] = data['race/ethnicity'].replace('group B', 2)
data['race/ethnicity'] = data['race/ethnicity'].replace('group C', 3)
data['race/ethnicity'] = data['race/ethnicity'].replace('group D', 4)
data['race/ethnicity'] = data['race/ethnicity'].replace('group E', 5)

# codificação de rótulo para o nível de educação dos pais
data['parental level of education'] = le_cod.fit_transform(data['parental level of education'])

# codificação de rótulo para gender
data['gender'] = le_cod.fit_transform(data['gender'])

# codificação de rótulo para "math score"
data['math score'] = le_cod.fit_transform(data['math score'])

# codificação de rótulo para "reading score"
data['reading score'] = le_cod.fit_transform(data['reading score'])

# codificação de rótulo para "writing score"
data['writing score'] = le_cod.fit_transform(data['writing score'])

data

x = data.iloc[:,:10]
y = data.iloc[:,10]

print(x.shape)
print(y.shape)

x

y

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 45)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train

y_train

x_test

y_test

# importando o MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# criando um escalonador
mm = MinMaxScaler()

# alimentando a variável independente no scaler
x_train = mm.fit_transform(x_train)
x_test = mm.transform(x_test)

from sklearn.decomposition import PCA

# creating a principal component analysis model
pca = PCA(n_components = None)

# alimentando as variáveis independentes para o modelo PCA
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# visualizar os principais componentes que explicarão a maior parcela de variância
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

# criando um modelo de análise de componentes principais
pca = PCA(n_components = 2)

# alimentando as variáveis independentes para o modelo PCA
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

from sklearn.linear_model import  LogisticRegression
import pickle

# criando um modelo
model = LogisticRegression()

# alimentando os dados de treinamento para o modelo
model.fit(x_train, y_train)

# prever os resultados do conjunto de teste
y_pred = model.predict(x_test)

# calcular as precisões de classificação
print("Precisão do treinamento :", model.score(x_train, y_train))
print("Teste de precisão :", model.score(x_test, y_test))

"""Salvar o modelo no **disco**"""

filename = 'finalized_model_log_reg.pkl'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)

# Aqui o teste com o mesmo modelo só que carregando o modelo do disco
print("Precisão do treinamento :", loaded_model.score(x_train, y_train))
print("Teste de precisão :", loaded_model.score(x_test, y_test))