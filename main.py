import random
import pandas as pd

class ia2():
    def __init__(self):
        self.db = pd.read_excel("Libro1.xlsx")

        self.network = {1: 3, 2: 3, 3: 3, 4: 2} #Red neuronal Clave=Capa Valor=Cantidad Neuronas.
        self.weightLayer = {} #Iniciar diccionario de capas de pesos.
        self.biases = {} #Diccionario de umbrales.
        self.forwardPropagationResults = {} #Resultado de las salidas por iteración {i: yi}
        self.newValueNeurons = {} #Nuevos resultados de las neuronas con las funciones de activación.
        self.batch = 5 #Lote
        self.doneBatch = {}

        self.weightGenerator()
        self.biasGenerator()
        self.forwardPropagation()

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

    def biasGenerator(self):
        for layer in self.network:
            self.biases[layer] = []
            if layer == 1:
                for neuron in range(1, self.network[layer]+1):
                    self.biases[layer].append([neuron, "No posee"])
            else:
                for neuron in range(1, self.network[layer]+1):
                    self.biases[layer].append([neuron, round(random.random(), 8)])
        
    def forwardPropagation(self):
        def transformValue(value, column):
            return round(((value - self.db[column].min()) / (self.db[column].max() - self.db[column].min())), 8)
        
        totalBatch = 0

        for batches in range(0, int(self.db.shape[0] / self.batch)+1):
            print(f'Batch número {batches} ->> Datos recorridos {totalBatch}')
            self.doneBatch[batches] = {}
            for neuronLayer in self.biases:
                self.newValueNeurons[neuronLayer] = []
                if neuronLayer == 1: #Verificar que sea la capa 1 para asignar los valores de entrada.
                    for index in range(totalBatch, self.batch):
                        pointer = 0
                        for column in self.db.columns[:len(self.biases[1])]:
                            value = self.db.iloc[index][column]
                            self.newValueNeurons[neuronLayer].append([pointer+1, transformValue(value, str(column))])
                            pointer +=1
                            print(self.newValueNeurons)
                        totalBatch += 1
                        
                else:
                    for neuron in self.biases[neuronLayer]:
                        print(neuron)
        print(self.doneBatch)
        
if __name__ == "__main__":
    ia2()