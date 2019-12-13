import numpy as np


class Individuo():
    def __init__(self, genes, base, classes, fitness=None):
        self.genes = genes
        self.fitness = fitness
        self.qtd_hidden = 10
        self.qtd_out = 2
        self.base = base
        self.classes = classes
        if fitness == None:
            self._reset_fit()

    def __str__(self):
        return str(self.fitness)  # +" "+str(self.genes)

    def __add__(self, other):
        return Individuo(self.genes + other.genes, self.base, self.classes, fitness= self.fitness + other.fitness)

    def __radd__(self, other):
        if 0 == other:
            return self
        else:
            return self.__add__(other)

    def mutation(self):
        indexes = np.arange(len(self.genes))
        np.random.shuffle(indexes)
        indexes = indexes[:len(self.genes) // 2]
        self.genes[indexes] = self.genes[indexes] + (np.random.random(len(self.genes) // 2) - 0.5)
        self._reset_fit()

    def _reset_fit(self):
        ## implementacao do predict de uma MLP
        y = self._predict(self.base)
        e = self.classes - np.argmax(y, axis=1)
        acc = len(e[e == 0].tolist()[0]) / self.base.shape[0]
        self.fitness = acc

    def _predict(self, base):
        X = np.hstack((np.full((base.shape[0], 1), -1), base))
        w = np.matrix(self.genes[:X.shape[1] * self.qtd_hidden]).reshape(
            (X.shape[1], self.qtd_hidden))
        m = np.matrix(self.genes[X.shape[1] * self.qtd_hidden:]).reshape(
            (w.shape[1] + 1, self.qtd_out))
        u1 = X * w
        h = 1 / (1 + np.exp(-u1))
        h = np.hstack((np.full((base.shape[0], 1), -1), h))

        u2 = h * m
        y = 1 / (1 + np.exp(-u2))
        return y


def fit(base, classes, n_pop, n_ger, mut_per, eli_per, cru_per):
    qtd_solucao = n_pop
    qtd_geracoes = n_ger
    perc_mutacoes = mut_per
    perc_elitismo = eli_per
    perc_cruzamento = cru_per

    ## para gerar quantidade de pesos da mlp
    qtd_neuronios_entrada = 2 + 1
    qtd_neuronios_escondida = 10
    qtd_neuronios_saida = 2
    qtd_pesos_mlp = (qtd_neuronios_entrada * qtd_neuronios_escondida) + (
            (qtd_neuronios_escondida + 1) * qtd_neuronios_saida)

    def fitness(cromossomo):
        ## implementacao do predict de uma MLP
        X = np.hstack((np.full((base.shape[0], 1), -1), base))
        w = np.matrix(cromossomo[:X.shape[1] * qtd_neuronios_escondida]).reshape((X.shape[1], qtd_neuronios_escondida))
        m = np.matrix(cromossomo[X.shape[1] * qtd_neuronios_escondida:]).reshape((w.shape[1] + 1, qtd_neuronios_saida))
        u1 = X * w
        h = 1 / (1 + np.exp(-u1))
        h = np.hstack((np.full((base.shape[0], 1), -1), h))

        u2 = h * m
        y = 1 / (1 + np.exp(-u2))
        e = np.matrix(classes).T - np.argmax(y, axis=1)
        acc = len(e[e == 0].tolist()[0]) / base.shape[0]
        return acc

    def cruzamento(pai1, pai2):
        ## cruzamento unico ponto
        ponto = np.random.randint(1, len(pai2.genes) - 1)
        filho1 = np.concatenate((pai1.genes[:ponto], pai2.genes[ponto:]), axis=None)
        filho2 = np.concatenate((pai2.genes[:ponto], pai1.genes[ponto:]), axis=None)
        return Individuo(filho1, base, classes), Individuo(filho2, base , classes)

    def selecao(populacao):
        soma = sum(populacao)
        r = np.random.random(1) * soma.fitness
        s = 0
        for i in populacao:
            s += i.fitness
            if s > r:
                return i
        pass

    def melhores(populacao):
        return populacao[:int(perc_elitismo * len(populacao))], populacao[int(perc_elitismo * len(populacao)):]
    bests = []
    ## criacao aleatoria da populacao
    populacao = np.random.random((qtd_solucao, qtd_pesos_mlp))
    pop = []
    winner = None
    for individuo in populacao:
        pop.append(Individuo(individuo, base, classes))
    pop.sort(key=lambda x: x.fitness, reverse=True)

    for k in range(qtd_geracoes):
        top, pop = melhores(pop)

        for i in range(int(len(populacao) * perc_cruzamento) // 2):
            pai1 = selecao(pop)
            pop.remove(pai1)
            pai2 = selecao(pop)
            pop.remove(pai2)
            filho1, filho2 = cruzamento(pai1, pai2)
            top.append(filho1)
            top.append(filho2)
        for i in pop:
            i.mutation()
            top.append(i)

        pop = top
        pop.sort(key=lambda x: x.fitness, reverse=True)
        bests.append(pop[0])
        if (pop[0].fitness == 1.0):
            break
    print(top[0].fitness)
    return top[0], bests
