# SVM_DIABETES
## Contexto
El código proporcionado se enfoca en la clasificación de datos de diabetes utilizando el algoritmo de Máquinas de Vectores de Soporte (SVM). El conjunto de datos utilizado es el conjunto de datos de diabetes incluido en la biblioteca sklearn.datasets. El objetivo es entrenar un modelo SVM para predecir si un paciente tiene diabetes o no, utilizando características relevantes del conjunto de datos.

## Objetivo
El objetivo principal es construir un modelo de clasificación preciso que pueda predecir con precisión la presencia de diabetes en un paciente dado cierto conjunto de características. Además, se busca realizar una exploración y análisis de los datos, normalizar los datos para mejorar el rendimiento del modelo y ajustar los hiperparámetros del modelo SVM para mejorar su rendimiento.

## Funcionamiento
El código comienza importando las bibliotecas necesarias y cargando el conjunto de datos de diabetes. Luego, realiza una exploración inicial de los datos, selecciona características relevantes, visualiza los datos, normaliza los datos, divide el conjunto de datos en conjuntos de entrenamiento y prueba, entrena un modelo SVM, evalúa el rendimiento del modelo y ajusta los hiperparámetros del modelo utilizando búsqueda en cuadrícula.

## Paquetes Necesarios para Instalar
* numpy: Para operaciones numéricas.
* matplotlib: Para visualización de datos.
* scikit-learn: Para implementar el modelo SVM, preprocesamiento de datos y evaluación del modelo.
* pandas: Para análisis de datos y manipulación de marcos de datos.
Puede instalar estas dependencias utilizando pip:

```python
pip install numpy matplotlib scikit-learn pandas
```
