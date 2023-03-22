# Face Recognition

![image](https://user-images.githubusercontent.com/76737266/227038931-7961793a-2842-464a-8dfb-d75eda3596ac.png)


## Introdução

Este programa tem a função de fazer a detecção e a identificação facial. Caso a pessoa esteja com cadastrada o algoritmo deve reconhecer a pessoa e mostrar o seu nome. Caso a pessoa não esteja no banco de dados ela deve ser classificada como desconhecida

## Instalação

para instalar os requisitos do programa o usuario deve entrar na pasta do programa e executar o segundo comando.

`pip install -r requirements.txt`

além disso é necessario que a maquina tenha uma camera para poder utilizar o reconhecimento em tempo real.

## Como utilizar

Para usar o reconhecimento em tempo real basta rodar o programa **Face_Recognition.py** para cadastrar um morador deve-se rodar o arquivo **Controle_Moradores.py** apertar a tecla *enter* e digitar o nome do novo morador, para retirar um morador da base de dados devemos apertar a tecla *9* e digitar o nome do morador.

## Conclusão

A rede neural se provou melhor que os outro métodos quanto o numero de moradores aumenta e em segundo lugar ficou o SVM com uma precisão bem satisfatória. Logo, a rede neural foi o unico método utlizado para rodar o programa em tempo real. Voce pode ver a comparação dos métodos no arquivo **Avaliando_Modelos.py.**






