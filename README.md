# Trabajo Práctico Final - Machine Learning
## Predicción de Accidentes Cerebrovasculares (ACV)

Este repositorio contiene el trabajo práctico final para la materia Machine Learning de la Carrera de Especialización en Inteligencia Artificial (CEIA) de la Universidad de Buenos Aires (UBA).

### Integrantes del Grupo
* Maximiliano Christener
* Ronald Uthurralt
* Juan Pablo Skobalski
* Luis Díaz

### Descripción del Proyecto
El accidente cerebrovascular (ACV) es una de las principales causas de muerte y discapacidad a nivel mundial. La detección temprana de pacientes con alto riesgo es crucial para la intervención clínica.

El objetivo de este trabajo es desarrollar y comparar modelos de aprendizaje automático capaces de predecir la probabilidad de que un paciente sufra un ACV basándose en parámetros clínicos y demográficos (edad, nivel de glucosa, IMC, hipertensión, etc.).

> Enfoque: Dada la naturaleza médica del problema, el enfoque principal se centra en minimizar los Falsos Negativos (evitar clasificar como "sanos" a pacientes que sufrirán un ACV).

### Dataset
Se utiliza el Stroke Prediction Dataset obtenido de Kaggle.

* Fuente: [Stroke Prediction Dataset - Federico Soriano](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
* Características: El set de datos contiene 11 atributos clínicos/demográficos y una variable objetivo binaria (`stroke`).
* Desafío Principal: El dataset presenta un desbalance de clases severo, con aproximadamente solo un 5% de casos positivos.

### Tecnologías y Librerías
El proyecto está desarrollado en Python 3.11+. Las principales librerías utilizadas son:

* Pandas & Numpy: Manipulación y análisis de datos.
* Scikit-Learn: Preprocesamiento, modelado y métricas.
* Imbalanced-Learn (imblearn): Técnicas de sobremuestreo (SMOTE) para manejo de clases desbalanceadas.
* CatBoost & XGBoost: Algoritmos de Gradient Boosting.
* Matplotlib & Seaborn: Visualización de datos.

### Metodología de Trabajo
El flujo de trabajo implementado en el notebook `tp_machine_learning-multi_model.ipynb` comprende:

1. Análisis Exploratorio de Datos (EDA): Estudio de distribuciones, correlaciones y detección de valores nulos (especialmente en BMI).
2. Preprocesamiento:
    * Imputación de datos faltantes (Estrategia: Mediana para numéricos, Frecuencia para categóricos).
    * Codificación de variables categóricas (One-Hot Encoding).
    * Escalado de variables numéricas.
3. Estrategia de Validación: División de datos en conjuntos de Entrenamiento, Validación y Test para asegurar una evaluación honesta y evitar fugas de datos (*data leakage*).
4. Manejo de Desbalance: Implementación de SMOTE (Synthetic Minority Over-sampling Technique) dentro de Pipelines y uso de pesos de clases (`class_weights`).
5. Modelado: Se entrenaron y optimizaron los siguientes modelos:
    * K-Nearest Neighbors (KNN)
    * Random Forest (RF)
    * XGBoost
    * CatBoost
6.  Optimización de Hiperparámetros: Uso de `RandomizedSearchCV` y `GridSearchCV`.
7.  Métrica de Selección: Se priorizó el F2-Score (que da más peso al *Recall* que a la *Precision*) para penalizar fuertemente los falsos negativos.

### Resultados y Conclusiones
Tras evaluar los distintos modelos y técnicas de calibración de probabilidades:

* Modelo Seleccionado: Random Forest (con ajuste de hiperparámetros y umbral de decisión optimizado).

Hallazgos principales:
* El modelo Random Forest ofreció el mejor equilibrio y la mayor sensibilidad (*Recall*) en el set de prueba, logrando detectar la mayor cantidad de casos positivos reales.
* Modelos como KNN mostraron un rendimiento inferior en la discriminación de la clase minoritaria.
* Se experimentó con técnicas de calibración de probabilidades (Sigmoid/Isotonic), pero se descartaron para el modelo final ya que reducían la sensibilidad en favor de la precisión, lo cual es contraproducente para este caso de uso médico.