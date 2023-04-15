import random
import pandas as pd

class ia2():
    def __init__(self):

        self.network = {1: 1, 2: 3, 3: 1} #Red neuronal Clave=Capa Valor=Cantidad Neuronas.
        self.weightLayer = {} #Iniciar diccionario de capas de pesos.
        self.biases = {} #Diccionario de umbrales.
        self.forwardPropagationResults = {} #Resultado de las salidas por iteración {i: yi}
        self.newValueNeurons = {} #Nuevos resultados de las neuronas con las funciones de activación.

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
        
        
if __name__ == "__main__":
    ia2()