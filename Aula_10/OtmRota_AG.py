# feito com auxilio de IA (Gemini)

import random
import numpy as np

cap_max = 8
vol = [2, 1, 3, 2, 1, 3, 1, 2, 3, 1, 2, 1]
num_b = 12

dist_mtx = [
    [0, 5, 9, 14, 7, 6, 12, 11, 8, 10, 13, 15],
    [5, 0, 4, 12, 6, 5, 11, 13, 9, 8, 14, 10],
    [9, 4, 0, 6, 10, 8, 12, 9, 7, 11, 13, 14],
    [14, 12, 6, 0, 8, 7, 9, 10, 12, 13, 5, 6],
    [7, 6, 10, 8, 0, 5, 8, 11, 10, 9, 12, 13],
    [6, 5, 8, 7, 5, 0, 6, 9, 8, 10, 11, 14],
    [12, 11, 12, 9, 8, 6, 0, 4, 7, 8, 10, 9],
    [11, 13, 9, 10, 11, 9, 4, 0, 3, 6, 7, 8],
    [8, 9, 7, 12, 10, 8, 7, 3, 0, 5, 9, 10],
    [10, 8, 11, 13, 9, 10, 8, 6, 5, 0, 4, 7],
    [13, 14, 13, 5, 12, 11, 10, 7, 9, 4, 0, 3],
    [15, 10, 14, 6, 13, 14, 9, 8, 10, 7, 3, 0]
]

def decod_rota(cromossomo):
    vgns = []
    vgm_atual = []
    cg_atual = 0
    
    for bairro in cromossomo:
        vol_bairro = vol[bairro]
        if cg_atual + vol_bairro <= cap_max:
            vgm_atual.append(bairro)
            cg_atual += vol_bairro
        else:
            vgns.append(vgm_atual)
            vgm_atual = [bairro]
            cg_atual = vol_bairro
    if vgm_atual:
        vgns.append(vgm_atual)
    return vgns

def calcular_fitness(cromossomo):
    vgns = decod_rota(cromossomo)
    dist_total = 0
    
    for viagem in vgns:
        if not viagem: continue
        dist_total += dist_mtx[0][viagem[0]]
        for i in range(len(viagem) - 1):
            dist_total += dist_mtx[viagem[i]][viagem[i+1]]
        dist_total += dist_mtx[viagem[-1]][0]
    
    return 1 / (dist_total + len(vgns) * 10)

def crossover(pai1, pai2):
    size = len(pai1)
    a, b = sorted(random.sample(range(size), 2))
    filho = [-1] * size
    filho[a:b] = pai1[a:b]
    
    p2_indc = [item for item in pai2 if item not in filho]
    pos = 0
    for i in range(size):
        if filho[i] == -1:
            filho[i] = p2_indc[pos]
            pos += 1
    return filho

def mutacao(cromossomo):
    idx1, idx2 = random.sample(range(len(cromossomo)), 2)
    cromossomo[idx1], cromossomo[idx2] = cromossomo[idx2], cromossomo[idx1]

def alg_gen(pop_size=50, geracoes=100):
    bairros_restantes = list(range(num_b))
    ppl = [random.sample(bairros_restantes, num_b) for _ in range(pop_size)]
    
    for gen in range(geracoes):
        ppl = sorted(ppl, key=lambda c: calcular_fitness(c), reverse=True)
        new_ppl = ppl[:10]
        
        while len(new_ppl) < pop_size:
            p1, p2 = random.sample(ppl[:20], 2)
            filho = crossover(p1, p2)
            if random.random() < 0.1: mutacao(filho)
            new_ppl.append(filho)
        ppl = new_ppl

    melhor = ppl[0]
    return melhor, decod_rota(melhor)

melhor_rota, vgns_finais = alg_gen()
print(f"melhor sequencia encontrada: {melhor_rota}")
print(f"divisao das vgns: {vgns_finais}")