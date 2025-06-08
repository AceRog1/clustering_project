# Proyecto 2: Clustering
Grupo 4

### Integrantes
* Toribio Arrieta, Josue Raul
* Rojas Urquizo, Andrés Alejandro
* Navarro Mendoza, Aaron Cesar Aldair
* Huarcaya Farfán, Dylan José

### Distribucion del codigo

* Dentro de la carpeta `/modelos` se puede encontrar los modelos usados para desarrollar el sistema de recomendacion y hacer sacar las metricas.
  * `main_script_test.py`: Estan los modelos de K-means y DBSCAN. Se uso principalmente para hacer pruebas con la data.
  * `main_script_fast_test.py`: Esta solo el modelo de k-means. Se uso principalmente para hacer pruebas mas rapidas, despues de ver que DBSCAN no nos fue util y demoraba mucho tiempo.
* Dentro de la carpeta `/notebooks` estan todos los archivos en el cual se trabajo principalmente el tratamiento de la data y generacion de CSVs para los modelos y la visualizacion.
  * `ingesta_y_preprocesamiento.ipynb`: Usado para sacar juntar y buscar las imagenes de los posters necesarias para el entrenamiento de los modelos de clustering.
  * `procesamiento_y_feature_engineering.ipynb`: Usado para extraer los features visuales (sin tener que descargar las imagenes), procesar estos datos y genrar los CSVs usados para el entrenamiento.
  * `test_processing.ipynb`: Usado para generar el dataset de test para los modelos de clustering ya entrenados. Aqui se realiza extraccion de features visuales, tratamiendo de la data, etc.
  * `clustering_test_sklearn.ipynb`: Usado para probar la que la extraccion de features haya sido correcta mediante el uso de librerias de SKlearn para el clustering.
  * `visualizacion.ipynb`: Usado para generar algunas graficas usadas en el informe.
* Dentro de la carpeta `/scripts`, se encuentran archivos adicionales usados en el proyecto.
  * `main_script_kaggle.py`: usado para generar los submission para kaggle.

### Enlaces para los Datasets usados en el proyecto

- https://drive.google.com/drive/folders/15mf1DzGpUmpHFHDbc-gFy5gY3JP-mM0z?usp=sharing 

