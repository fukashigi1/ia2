import random

#Red neuronal fija 1 input, 1 capa de 3 neuronas y una salida.

class ia2():
    def __init__(self):
        self.network = {1: 1, 2: 3, 3: 1} #Red neuronal Clave=Capa Valor=Cantidad Neuronas.
        self.weightLayer = {} #Iniciar diccionario de capas de pesos.
        self.biases = {} #Diccionario de umbrales.
        self.used = [] #Chequea que ya se ha usado ese key
        self.newValuesNeurons = {} #Lugar para almacenar las neuronas con su activación.
        self.data = {} #database
        self.dataList = [] #keys a lista.
        self.finalsY = {}
        self.batch = 5
        self.learnRate = 0.1
        self.batchCount = {}
        self.gradientDescend = {} #key: batch, value: [new weigts]

        self.dataBaseGenerator()
        self.dataToList()
        self.weightGenerator()
        self.biasGenerator()

        self.batchPointer = 0
        for i in range(len(self.data)):#lotes lotes
            if (i+1) % self.batch == 0:
                print(f'Batch {self.batchPointer + 1} -----------------------------------------')
                dicvalues = list(self.data.values())
                
                for z in range(i - 4, (i - 4) + self.batch):
                    self.finalsY[i] = self.forwardPropagation() 
                    self.batchCount[z] = self.backPropagation(dicvalues[z], self.finalsY[i])
                #falta promediar los batches y cambiar los pesos cada 5 iteraciones. (batchs)

                _0 = 0
                _1 = 0
                _2 = 0
                _3 = 0
                _4 = 0
                _5 = 0

                for x in range(i - 4, (i - 4) + self.batch):
                    pointer = 0
                    for index in self.batchCount[x]:
                        if pointer == 0:
                            _0 = _0 + index[0]
                        if pointer == 1:
                            _1 = _1 + index[0]
                        if pointer == 2:
                            _2 = _2 + index[0]
                        if pointer == 3:
                            _3 = _3 + index[0]
                        if pointer == 4:
                            _4 = _4 + index[0]
                        if pointer == 5:
                            _5 = _5 + index[0]
                        pointer += 1
                    
                _0 = _0 / self.batch
                _1 = _1 / self.batch
                _2 = _2 / self.batch
                _3 = _3 / self.batch
                _4 = _4 / self.batch
                _5 = _5 / self.batch
                print(_0)
                print(_1)
                print(_2)
                print(_3)
                print(_4)
                print(_5)
                print("----------------")
                #recorrer el for al reves self.weightlayer
                for layer in reversed(self.weightLayer):
                    pointer = 2
                    for weight in reversed(self.weightLayer[layer]):
                        print(weight)
                        if pointer == 2 and layer == 2:
                           self.weightLayer[layer][pointer] = [round(_0, 8), 3, 1]
                        if pointer == 1 and layer == 2:
                            self.weightLayer[layer][pointer] = [round(_1, 8), 2, 1]
                        if pointer == 0 and layer == 2:
                            self.weightLayer[layer][pointer] = [round(_2, 8), 1, 1]
                        if pointer == 2 and layer == 1:
                            self.weightLayer[layer][pointer] = [round(_3, 8), 1, 3]
                        if pointer == 1 and layer == 1:
                            self.weightLayer[layer][pointer] = [round(_4, 8), 1, 2]
                        if pointer == 0 and layer == 1:
                            self.weightLayer[layer][pointer] = [round(_5, 8), 1, 1]
                        pointer -=1
                self.batchPointer += 1

    def weightGenerator(self):
        for layer in range(1, len(self.network)):
            self.weightLayer[layer] = []
            weightsInLayer = self.network[layer] * self.network[layer+1]
            for weights in range(1, weightsInLayer+1):
                self.weightLayer[layer].append([round(random.random(), 8)])
            pointer = 0
            for i in range(1, self.network[layer] + 1):
                for j in range(1, self.network[layer+1] + 1):
                        self.weightLayer[layer][pointer].append(i)
                        self.weightLayer[layer][pointer].append(j)
                        pointer = pointer +1
        print(self.weightLayer)
    def biasGenerator(self):
        for layer in self.network:
            self.biases[layer] = []
            if layer == 1:
                for neuron in range(1, self.network[layer]+1):
                    self.biases[layer].append([neuron, "No posee"])
            else:
                for neuron in range(1, self.network[layer]+1):
                    self.biases[layer].append([neuron, round(random.random(), 8)])
        
    def dataBaseGenerator(self):
        for x in range(0, 100):
            while True:
                clave = random.randint(-1000, 1000)
                if clave not in self.data:
                    break
            valor = 1 if clave >= 0 else 0
            self.data[clave] = valor

    def dataToList(self): #Este metodo transforma el diccionario self.data en una lista que almacena los valores de X
        for key in self.data:
            self.dataList.append(key)

    def forwardPropagation(self):
        self.newValuesNeurons.clear()
        def sigmoidActivation(value):
            return 1 / (1 + pow(2.71828, -value))

        def inputValue(value):
            convertedValue = (value - min(self.dataList))/(max(self.dataList) - min(self.dataList))
            return round(convertedValue, 8)

        for layer in self.biases:
            self.newValuesNeurons[layer] = []
            for neuron in self.biases[layer]:
                if layer == 1:
                    for key, value in self.data.items():
                        if key in self.used:
                            pass
                        else:
                            self.newValuesNeurons[layer].append([neuron[0], inputValue(key)])
                            self.used.append(key)
                            break

                if layer == 2:
                    for weightLayer in self.weightLayer:
                        for weight in self.weightLayer[weightLayer]:
                            if weightLayer == layer-1 and weight[2] == neuron[0]:
                                self.newValuesNeurons[layer].append([neuron[0], round(sigmoidActivation((weight[0]*self.newValuesNeurons[1][0][1]) + neuron[1]), 8)])

                if layer == 3:
                    fx=[] #multiplication
                    for weightLayer in self.weightLayer:
                        for weight in self.weightLayer[weightLayer]:
                            if weightLayer == layer-1 and weight[2] == neuron[0]:
                                for stepBackNeuron in self.newValuesNeurons[layer - 1]:
                                    if stepBackNeuron[0] == weight[1]:
                                        fx.append(round(weight[0]*stepBackNeuron[1], 8))
                    sumatory = 0
                    for value in fx:
                        sumatory += value
                    self.newValuesNeurons[layer].append([neuron[0], round(sigmoidActivation(sumatory + neuron[1]), 8)])
        return self.newValuesNeurons[3][0][1]
    
    def backPropagation(self, S, Y):
        def convertValue(value):
            return round(value * (1 - value), 8)
        
        gradientDescend = [] 
        dedy = -(S - Y)
        for layer in reversed(self.weightLayer):
            for weight in reversed(self.weightLayer[layer]):
                ecuation = []
                if layer == 2:
                    ecuation.append(convertValue(Y))
                    for neuron in self.newValuesNeurons[2]:
                        if weight[1] == neuron[0]:
                            ecuation.append(neuron[1])
                    dydw = 1
                    for multiplication in ecuation:
                        dydw = dydw * multiplication
                    dedw = dedy * dydw
                    newWeight = weight[0] - self.learnRate * dedw
                    gradientDescend.append([round(newWeight, 8), weight[1], weight[2]])
                if layer == 1:
                    ecuation.append(convertValue(Y))
                    for neuron in self.newValuesNeurons[1]:
                        ecuation.append(neuron[1])
                    for neuron in self.newValuesNeurons[2]:
                        if weight[2] == neuron[0]:
                            ecuation.append(neuron[1])
                            for otherWeight in self.weightLayer[2]:
                                if otherWeight[1] == neuron[0]:
                                    ecuation.append(otherWeight[0])
                    dydw = 1
                    for multiplication in ecuation:
                        dydw = dydw * multiplication
                    dedw = dedy * dydw
                    newWeight = weight[0] - self.learnRate * dedw
                    gradientDescend.append([round(newWeight, 8), weight[1], weight[2]])
        return gradientDescend
    
            
if __name__ == "__main__":
    ia2()