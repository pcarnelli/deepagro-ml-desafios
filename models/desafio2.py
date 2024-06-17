# -*- coding: utf-8 -*-
"""  Implementación de outlier pooling en ResNet-18.

En este script se implementa, usando PyTorch, la técnica de outlier pooling
en una red profunda ResNet-18. El outlier pooling reemplaza al average pooling
global usualmente utilizado en ResNet.

Se usan como base las capas y bloques de ResNet-18 predefinidas en PyTorch y se
crea la clase ResNet18OutlierPooling. Toma como entrada un tensor de tamaño
(n_lote, n_canales, alto, ancho). Donde n_lote es el cantidad de imágenes por
lote usadas para entrenar/evaluar el modelo, n_canales es la cantidad de canales
de la imagen original y (alto, ancho) son las dimensiones de cada imagen.

Los siguientes paquetes son necesarios: torch, torchvision.

Contiene las siguientes clases:

    * ResNet18OutlierPooling - Red profunda ResNet-18 con outlier pooling.


@author: Patricio Carnelli
@contact: pcarnelli@gmail.com
@credit: Chao Ren et al
@links: https://doi.org/10.1109/CVPRW53098.2021.00328
@date: 16-Jun-2024
@version: 0.1
"""


import torch
from torchvision import models


class ResNet18OutlierPooling(torch.nn.Module):
    """Construye una red profunda ResNet-18 con outlier pooling (subclase de
    torch.nn.Module).

    Attrs:
        n_clases (int): Cantidad de clases a predecir.
        param_umbral (float): Parámetro del umbral de outlier pooling (lambda).
    """


    def __init__(self, n_clases: int, param_umbral: float):
        """Constructor de la clase ResNet18OutlierPooling.

        Args:
            n_clases (int): Cantidad de clases a predecir.
            param_umbral (float): Parámetro del umbral de outlier pooling (lambda).
        """

        super().__init__()
        self.n_clases = n_clases
        self.param_umbral = param_umbral

    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Define el cálculo que se hace en cada llamada a la clase.

        Args:
            X (torch.Tensor): Entrada (n_lote, n_canales, alto, ancho).

        Returns:
            torch.Tensor: Resultado (n_lote, n_clases, 1, 1).
        """

        Y = self.red_base(X)
        Y = self.outlier_pooling(Y)
        Y = self.capa_densa(Y)

        return Y


    def red_base(self, X: torch.Tensor) -> torch.Tensor:
        """Aplica todos los bloques de ResNet-18 hasta antes del average pooling.

        Args:
            X (torch.Tensor): Entrada (n_lote, n_canales, alto, ancho).

        Returns:
            torch.Tensor: Resultado (n_lote, 512, techo(alto/32), techo(ancho/32)).
        """

        resnet18 = models.resnet18()
        red = torch.nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,
        )

        return red(X)


    def outlier_pooling(self, X: torch.Tensor) -> torch.Tensor:
        """Aplica el método de outlier pooling.

        Args:
            X (torch.Tensor): Entrada (n_lote, 512, techo(alto/32), techo(ancho/32)).

        Returns:
            torch.Tensor: Resultado (n_lote, 512, 1, 1).
        """

        # Redimensiona el tensor de entrada para el cálculo del promedio y
        # el desvío estándar necesarios para obtener el valor del umbral
        X = X.reshape(X.size()[0], X.size()[1], X.size()[2]*X.size()[3])

        umbral = torch.mean(X, -1, keepdim=True)
        umbral = umbral + self.param_umbral * torch.std(X, -1, keepdim=True)
        es_mayor_igual_umbral = (X >= umbral)
        
        Y = torch.sum(es_mayor_igual_umbral * X, -1)
        Y = Y / (torch.sum(es_mayor_igual_umbral, -1) + 0.001) # Suma 0.001 para evitar NaNs
        
        return Y
    

    def capa_densa(self, X: torch.Tensor) -> torch.Tensor:
        """Aplica la capa densa final de la red profunda ResNet-18.

        Args:
            X (torch.Tensor): Entrada (n_lote, 512, 1, 1).

        Returns:
            torch.Tensor: Resultado (n_lote, n_clases, 1, 1).
        """
        
        # Redimensiona el tensor de entrada (equivalente a torch.nn.Flatten)
        X = X.view(X.size()[0], -1)

        cd = torch.nn.Linear(512, self.n_clases)

        return cd(X)
    