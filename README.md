# DeepAgro - Machine Learning Engineer - Desafíos

## Desafío 1

El primer desafío consiste en evaluar cuatro modelos de clasificación de malezas para ver cuál de ellos poner en producción. Los detalles sobre el desafío están en el archivo `Enunciado 1.pdf`, ubicado en el directorio [resources](./resources/).

Para este desafío creé un notebook de Jupyter, llamado `desafio1.ipynb`, ubicado en el directorio [notebooks](./notebooks/). Estas son las opciones para visualizar o ejecutar dicho notebook:

1) Visualizar en github: La versión subida a github es el resultado de ejecutar el notebook en mi computadora local. Puede visualizarse directamente desde github. (Esta opción tiene el inconveniente de que el reporte del paquete `ydata-profiling` no se muestra en github. Ver la solución alternativa explicada en el mismo notebook.)

2) Ejecutar en un entorno virtual: Para ejecutar el notebook, el usuario puede crear un entorno virtual de CONDA usando el archivo `environment.yml`. O usando `venv` (con Python 3.12.3) e instalando las dependencias del archivo `requirements.txt` con `pip`. Ambos archivos, `environment.yml` y `requirements.txt`, están ubicados en el directorio [notebooks](./notebooks/).

3) Visualizar/ejecutar en Google Colab: Una versión ya ejecutada del notebook está disponible [aquí](https://colab.research.google.com/github/pcarnelli/deepagro-ml-desafios/blob/main/notebooks/desafio1_colab.ipynb). Esta versión se puede volver a ejecutar si el usuario lo desea, para lo que deberá registrarse con una cuenta de Google.

## Desafío 2

En el segundo desafío hay que implementar el método de "outlier pooling" en una red profunda (por ejemplo, ResNet-18). Se pueden encontrar más detalles sobre el desafío en los archivos `Enunciado 2.pdf` y `Ren_et_al_2021.pdf`, ambos ubicados en el directorio [resources](./resources/).

Mi propuesta de implementación e integración con ResNet-18 están en el script `desafio2.py`, ubicado en el directorio [models](./models/).
