import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv")
data.head()

data.describe()

print("cantidad de filas y columnas", data.shape)
print("Nombre de las columnas",data.columns)
data.info()

#Paso 2: Realiza un EDA completo

#Analisis de variables numericas
# Paso 1: Obtener columnas numéricas
var = data.select_dtypes(include='number').columns.tolist()

# Paso 2: Definir dimensiones fijas del grid
filas = 2
columnas = 5

# Paso 3: Crear subplots
fig, axes = plt.subplots(filas, columnas, figsize=(20, 8))  # Ajusta tamaño como quieras

axes = axes.flatten()  # Aplanar para acceso fácil

# Paso 4: Graficar cada variable
for i, col in enumerate(var):
    if i < len(axes):  # Para no pasarte del número de axes
        sns.histplot(data=data, x=col, ax=axes[i])
        axes[i].set_title(f"Histograma de {col}")

# Si sobran subplots, los apagas
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#Analisis de variables categoricas

# Visualización de variables categóricas

# Paso 1: Obtener columnas numéricas
var = data.select_dtypes(include='object').columns.tolist()

# Paso 2: Definir dimensiones fijas del grid
filas = 2
columnas = 5

# Paso 3: Crear subplots
fig, axes = plt.subplots(filas, columnas, figsize=(20, 8))  # Ajusta tamaño como quieras

axes = axes.flatten()  # Aplanar para acceso fácil

# Paso 4: Graficar cada variable
for i, col in enumerate(var):
    if i < len(axes):  # Para no pasarte del número de axes
        sns.histplot(data=data, x=col, ax=axes[i])
        axes[i].set_title(f"Histograma de {col}")

# Si sobran subplots, los apagas
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

fig, axis = plt.subplots(1, 3, figsize=(18, 6))  # Una fila, tres columnas

# Sexo vs Charges
sns.boxplot(ax=axis[0], data=data, x="sex", y="charges")
axis[0].set_title("Gastos médicos por sexo")

# Fumador vs Charges
sns.boxplot(ax=axis[1], data=data, x="smoker", y="charges")
axis[1].set_title("Gastos médicos por hábito de fumar")

# Región vs Charges
sns.boxplot(ax=axis[2], data=data, x="region", y="charges")
axis[2].set_title("Gastos médicos por región")

plt.tight_layout()
plt.show()

#Matriz de correlacion de variables numericas.
import seaborn as sns
import matplotlib.pyplot as plt

datacore = data.copy()
# Seleccionamos solo las columnas numéricas
num_data = datacore.select_dtypes(include='number')

# Calculamos la matriz de correlación
corr_matrix = num_data.corr()

# Creamos el heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Matriz de correlación entre variables numéricas")
plt.show()

#Maztriz de correlacion variables numericas y categoricas.

datacore.columns

datacore["sex"]  = pd.factorize(datacore["sex"])[0]
datacore["smoker"]  = pd.factorize(datacore["smoker"])[0]
datacore["region"]  = pd.factorize(datacore["region"])[0]


# Ver el resultado

fig, axes = plt.subplots(figsize=(15, 15))

sns.heatmap(datacore[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

# Draw Plot
plt.show()


#Eliminamos variables que carecen de inportancia en nuestro analisis
datafinal = data.copy()
# Eliminar columnas solo si existen
columns_to_drop = ["children", "sex","region"]
columns_to_drop = [col for col in columns_to_drop if col in datafinal.columns]  # Filtrar solo las que existen
datafinal= datafinal.drop(columns=columns_to_drop)

print(datafinal.columns)
datafinal

data2 = data.copy()

# Asegúrate de usar el mismo DataFrame
data2["smoker"] = data2["smoker"].map({"yes": 1, "no": 0})  # O ya debería estar hecho

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Gráfico 1: age vs charges
sns.regplot(ax=axes[0, 0], data=data2, x="age", y="charges")
sns.heatmap(data2[["charges", "age"]].corr(), annot=True, fmt=".2f", ax=axes[1, 0], cbar=False)

# Gráfico 2: bmi vs charges
sns.regplot(ax=axes[0, 1], data=data2, x="bmi", y="charges")
sns.heatmap(data2[["charges", "bmi"]].corr(), annot=True, fmt=".2f", ax=axes[1, 1], cbar=False)

# Gráfico 3: smoker vs charges
sns.regplot(ax=axes[0, 2], data=data2, x="smoker", y="charges")
sns.heatmap(data2[["charges", "smoker"]].corr(), annot=True, fmt=".2f", ax=axes[1, 2], cbar=False)

plt.tight_layout()
plt.show()

#Descripcion de columnas

# Paso 1: Columnas
columnas = datafinal.columns
n = len(columnas)

# Aquí guardaremos las descripciones como diccionario
descripciones = {}

# Paso 2: Definir grid
filas = (n + 7) // 8  # 8 columnas por fila
fig, axes = plt.subplots(filas, 8, figsize=(20, 3 * filas))
axes = axes.flatten()

# Paso 3: Guardar descripciones y graficarlas
for i, col in enumerate(columnas):
    descripcion = datafinal[col].describe()
    
    # Guardamos en el diccionario
    descripciones[col] = descripcion
    
    # Convertimos a texto para graficarlo
    texto = descripcion.to_string()

    axes[i].axis('off')
    axes[i].text(0, 1, texto, fontsize=9, va='top', ha='left', family='monospace')
    axes[i].set_title(f"Descripción: {col}", fontsize=10, pad=10)

# Apagamos los subplots vacíos
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()

#Limpieza de outlers por columnas

# Crear una copia de `datafinal` para limpiarlo sin modificar el original
datafinal_limpio = datafinal.copy()

# Iteramos sobre las columnas en el diccionario de descripciones
for columna in descripciones:
    descripcion = descripciones[columna]

    # Revisamos que existan los cuartiles
    if "75%" in descripcion.index and "25%" in descripcion.index:
        rango_iqr = descripcion["75%"] - descripcion["25%"]
        limite_superior = descripcion["75%"] + 1.5 * rango_iqr
        limite_inferior = descripcion["25%"] - 1.5 * rango_iqr

        # Poner NaN donde haya outliers
        datafinal_limpio.loc[(datafinal_limpio[columna] < limite_inferior) | (datafinal_limpio[columna] > limite_superior), columna] = np.nan
    else:
        print(f"Saltando columna {columna}: es una variable categórica")

# Eliminar las filas con al menos un NaN (outliers)
datafinal_limpio = datafinal_limpio.dropna()
datosdemodelo= datafinal_limpio
datosdemodelo2= datafinal_limpio
# Mostrar el dataset limpio
datafinal_limpio.head()

print(datafinal.shape, " VS ", datafinal_limpio.shape)

#Paso 3: Construir un modelo de regresion lineal 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Variable objetivo
y = datafinal_limpio["charges"]
# Variables predictoras
X = datafinal_limpio.drop(["charges"], axis=1)

#Convertimos columnas en 0s y 1s
X = pd.get_dummies(X, columns=['smoker'])

# Separar en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#Entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)

print(f"Intercepto (a): {model.intercept_}")
print(f"Coeficientes (b1, b2): {model.coef_}")

y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import mean_squared_error, r2_score

print(f"Error cuadrático medio: {mean_squared_error(y_test, y_pred)}")
print(f"Coeficiente de determinación: {r2_score(y_test, y_pred)}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # línea ideal
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Ajuste del modelo de regresión lineal")
plt.tight_layout()
plt.show()

#Segundo modelo sin limpiar outlers

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Variable objetivo
y = datafinal["charges"]
# Variables predictoras
X = datafinal.drop(["charges"], axis=1)

#Convertimos columnas en 0s y 1s
X = pd.get_dummies(X, columns=['smoker'])

# Separar en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#Entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

print(f"Intercepto (a): {model.intercept_}")
print(f"Coeficientes (b1, b2): {model.coef_}")

from sklearn.metrics import mean_squared_error, r2_score

print(f"Error cuadrático medio: {mean_squared_error(y_test, y_pred)}")
print(f"Coeficiente de determinación: {r2_score(y_test, y_pred)}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # línea ideal
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Ajuste del modelo de regresión lineal")
plt.tight_layout()
plt.show()

#RMSE más bajo con outliers eliminados sugiere que el modelo se ajusta mejor en términos absolutos a los datos limpios (predice más cerca del valor real).

#R² más alto con outliers presentes indica que el modelo explica más varianza cuando los outliers están incluidos. Pero esto no necesariamente es bueno: los outliers pueden estar sesgando el modelo y dándole una falsa sensación de “mejor ajuste”.
