import tf, { train } from '@tensorflow/tfjs';


async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    // Primeira camada: 7 entradas (idade, cores, localizações)
    // 80 neurônios para maior capacidade de aprendizado
    // ReLU filtra valores negativos, passando só informações relevantes
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    //saida precisa de 3 neuronios pois sao 3 categorias a serem previstas (premium, medium, basic)
    //activation 'softmax' é a função de ativação mais comum para classificação multiclasse, pois ela converte os valores de saída em probabilidades que somam 1, facilitando a interpretação dos resultados.
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));


    //compilar o modelo é o processo de configurar o modelo para treinamento, especificando o otimizador, a função de perda e as métricas que serão usadas para avaliar o desempenho do modelo durante o treinamento.
    model.compile({
        optimizer: 'adam', // algoritmo de otimização para ajustar os pesos da rede
        loss: 'categoricalCrossentropy', // função de perda para classificação multiclasse
        metrics: ['accuracy'] // métrica para avaliar o desempenho do modelo durante o treinamento
    });

    //treinamento do modelo, onde o modelo é ajustado para aprender a partir dos dados de entrada (inputXs) e das saídas esperadas (outputYs). O processo de treinamento envolve a iteração sobre os dados, ajustando os pesos da rede neural para minimizar a função de perda.
    await model.fit(inputXs, outputYs, {
        verbose: 0, // para não exibir o progresso do treinamento no console
        epochs: 100, // número de épocas de treinamento
        shuffle: true, // embaralha os dados a cada época para melhorar o treinamento e evitar vies
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Época ${epoch + 1}: perda = ${logs.loss.toFixed(4)}, precisão = ${logs.acc.toFixed(4)}`);
            } // comentei pra evitar log chato durante o treinamento, mas pode ser útil para acompanhar o progresso do modelo
        }
    });

    return model;
}

async function predict(model, inputTensor) {

    const tfInput = tf.tensor2d(inputTensor);
    const prediction = model.predict(tfInput);

    const predArray = await prediction.array();

    return predArray[0].map((prob,index) =>({prob,index}));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

inputXs.print();
outputYs.print();

const model = await trainModel(inputXs, outputYs);

const pessoasTeste = { nome: "Zé", idade: 28, cor: "verde", localizacao: "Curitiba" };
// normalizando os dados de teste
const pessoaTensorNormalizado = [
    [
        0.2, // idade normalizada
        1,    // cor azul
        0,    // cor vermelho
        0,    // cor verde
        1,    // localização São Paulo
        0,    // localização Rio
        0     // localização Curitiba
    ]
];

const predictions = await predict(model, pessoaTensorNormalizado);

const resultados = predictions.sort((a, b) => b.prob - a.prob).map(pred => ({
    categoria: labelsNomes[pred.index],
    probabilidade: pred.prob
}));

console.log("Previsões para Zé:");
resultados.forEach(result => {
    console.log(`${result.categoria}: ${(result.probabilidade * 100).toFixed(2)}%`);
});

