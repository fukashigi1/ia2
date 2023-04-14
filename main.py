import random
class ia2():
    def __init__(self):
        self.network = {1: 2, 2: 3, 3: 3, 4: 2} #Red neuronal Clave=Capa Valor=Cantidad Neuronas.
        self.weightLayer = {} #Iniciar diccionario de capas de pesos.
        self.biases = {} #Diccionario de umbrales.
        self.XS = [] #Base de datos con X y S. Si los dos son positivos dará 1, de lo contrario dará 0.

        self.batch = 2

        self.weightGenerator()
        self.biasGenerator()
        self.dataBaseGenerator()
        self.inputValuesGenerator()

    def weightGenerator(self):
        for layer in range(1, len(self.network)):
            self.weightLayer[layer] = []
            weightsInLayer = self.network[layer] * self.network[layer+1]
            for ws in range(1, weightsInLayer+1):
                self.weightLayer[layer].append(random.random())

    def biasGenerator(self):
        for layer in self.network:
            self.biases[layer] = []
            if layer == 1:
                pass
            else:
                for neuron in range(1, self.network[layer]+1):
                    self.biases[layer].append(random.random())

    def dataBaseGenerator(self):
        for i in range (0, 20):
            row = []
            for neuron in range(1, self.network[1]+1):
                key = random.randint(-100, 100)
                row.append(key)
            if all(value > 0 for value in row):
                val = 1
            else:
                val = 0
            row.append(val)
            self.XS.append(row)
    
    def inputValuesGenerator(self, iteration):
        for z in range(0, iteration):
            for neurons in range(1, self.network[1]+1):
                pass 
            #el entrenamiento en lotes acumula los gradientes de error para un conjunto de datos completo y luego realiza una actualización de pesos única basada en esos gradientes acumulados.

if __name__ == "__main__":
    ia2()