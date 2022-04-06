import random
import copy

class Perceptron(object):
    def __init__(self, samples, outputs, learning_rate=0.1, epochs=1000, limiar=-1):
        self.sample = samples                   #todas as amostras
        self.output = outputs                   #saidas de cada amostra
        self.learning_rate = learning_rate      #taxa de aprendizado
        self.epochs = epochs                    #numero de epocas
        self.limiar = limiar                    #limiar
        self.weights = []                       #vetor de pesos
        self.len_sample = len(samples[0])       #tamanho do vetor de amostras
        self.len_samples = len(samples)         #tamanho do vetor de amostras

    def train(self):
        #adiciona -1 para cada amostra
        for sample in self.sample:
            sample.insert(0, -1)
        
        #adiciona pesos aleatorios
        for i in range(self.len_sample):
            self.weights.append(random.uniform(-1, 1))
        
        #insere o limiar no vetor de pesos
        self.weights.insert(0, self.limiar)

        #iniciar o contador de epocas
        num_epochs = 0

        while True:
            #iniciar o contador de erros
            num_errors = 0
            for i in range(self.len_samples):
                #calcular a saida da rede
                output = self.calc_output(self.sample[i])
                #calcular o erro
                error = self.output[i] - output
                #atualizar os pesos
                self.update_weights(self.sample[i], error)
                #contar o erro
                if error != 0:
                    num_errors += 1
            #contar a epoca
            num_epochs += 1
            #parar se nao houver mais erros
            if num_errors == 0 or num_epochs >= self.epochs:
                break
    
    def calc_output(self, sample):
        #calcular a saida da rede
        output = 0
        for i in range(self.len_sample):
            output += sample[i] * self.weights[i]
        #retornar a saida da rede
        return self.activation(output)

    
    def activation(self, output):
        #retornar 1 se o output for maior que 0
        if output > 0:
            return 1
        #retornar 0 se o output for menor ou igual a 0
        else:
            return 0

    def update_weights(self, sample, error):
        #atualizar os pesos
        for i in range(self.len_sample):
            self.weights[i] += self.learning_rate * error * sample[i]
    
    def test(self, sample):
        #adiciona -1 para cada amostra
        sample.insert(0, -1)
        #calcular a saida da rede
        output = self.calc_output(sample)
        #retornar a saida da rede
        return output



samples = [[0.1, 0.4, 0.7], [0.3, 0.7, 0.2], [0.6, 0.9, 0.8], [0.5, 0.7, 0.1]]
outputs = [1, 1, -1, 1]

test = copy.deepcopy(samples)

perceptron = Perceptron(samples, outputs)
perceptron.train()

for i in test:
    print(perceptron.test(i))