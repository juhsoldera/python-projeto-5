import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.datasets import load_boston

#ignorando os warnings nan
np.warnings.filterwarnings('ignore')

#carregando a base
boston_dataset = load_boston()
boston_df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston_df['MEDV'] = boston_dataset.target
df_reduzido = pd.DataFrame()

#apos plotar o distplot de todos os parametros e eliminar os que menos faziam sentido...
df_reduzido['RM'] = boston_df['RM']
df_reduzido['B'] = boston_df['B']
df_reduzido['LSTAT'] = boston_df['LSTAT']
df_reduzido['MEDV'] = boston_df['MEDV']

#1 - head, info, describe
print("\nHead:")
print(boston_df.head())
print("\nInfo:")
print(boston_df.info())
print("\nDescribe:")
print(boston_df.describe())

#2 - Mostre o pairplot e escolha duas colunas cuja correlação faça sentido
sns.pairplot(df_reduzido)
pp.show()

#3 - Mostre sua escolha por meio de dois jointplots (seaborn):
#3.1 - scatter (default)
sns.jointplot(x="RM", y="MEDV", data=boston_df)
pp.show()

#3.2 - heatmap(hex)
sns.jointplot("RM", "MEDV", data=boston_df, kind="hex")
pp.show()

X = boston_df[['RM']]
y = boston_df['MEDV']

#4 - Realize a divisão, treinamento e predição
#4.1 - dividindo a base em treino e teste
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5) #random state = semente
lm = LinearRegression()

#4.2 - executando o treinamento
lm.fit(x_train, y_train)

#4.3 - executando a predicao
predicoes = lm.predict(x_test)

#5 - Mostre o resultado por meio de um
#5.1 - scatter
pp.scatter(y_test, predicoes)
pp.show()

#5.2 - e de um distplot
sns.distplot(y_test - predicoes, bins=10)
pp.show()