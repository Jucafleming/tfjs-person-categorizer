# People Classifier with TensorFlow.js

This project demonstrates how to use a simple neural network in JavaScript, using TensorFlow.js, to classify people into categories (premium, medium, basic) based on features such as age, favorite color, and location.

## Features
- Training a multiclass classification model
- Data preprocessing (normalization and one-hot encoding)
- Usage example with three fictional people
- Printing input and output tensors
- Displaying training progress in the console

## How it works
The model receives as input a vector with 7 values:
- Normalized age
- Favorite color (blue, red, green) — one-hot encoded
- Location (São Paulo, Rio, Curitiba) — one-hot encoded

The output is a classification into one of three categories: premium, medium, or basic (also one-hot encoded).

## Input example
```js
// Order: [normalized_age, blue, red, green, São Paulo, Rio, Curitiba]
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
];
```

## How to run
1. Install dependencies:
   ```powershell
   npm install @tensorflow/tfjs
   ```
2. Run the script:
   ```powershell
   node index.js
   ```

## Requirements
- Node.js
- npm

## Author
- Your Name

---
This project is just an educational example to demonstrate basic classification concepts with neural networks in JavaScript.

---

# Classificador de Pessoas com TensorFlow.js

Este projeto demonstra como utilizar uma rede neural simples em JavaScript, usando TensorFlow.js, para classificar pessoas em categorias (premium, medium, basic) com base em características como idade, cor favorita e localização.

## Funcionalidades
- Treinamento de um modelo de classificação multiclasse
- Pré-processamento dos dados (normalização e one-hot encoding)
- Exemplo de uso com três pessoas fictícias
- Impressão dos tensores de entrada e saída
- Exibição do progresso do treinamento no console

## Como funciona
O modelo recebe como entrada um vetor com 7 valores:
- Idade normalizada
- Cor favorita (azul, vermelho, verde) — one-hot encoded
- Localização (São Paulo, Rio, Curitiba) — one-hot encoded

A saída é uma classificação em uma das três categorias: premium, medium ou basic (também one-hot encoded).

## Exemplo de entrada
```js
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
];
```

## Como rodar
1. Instale as dependências:
   ```powershell
   npm install @tensorflow/tfjs
   ```
2. Execute o script:
   ```powershell
   node index.js
   ```

## Requisitos
- Node.js
- npm

## Autor
- Seu Nome

---
Este projeto é apenas um exemplo educacional para demonstrar conceitos básicos de classificação com redes neurais em JavaScript.
