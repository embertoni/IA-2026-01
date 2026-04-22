# feito com auxílio de IA (Gemini)

import heapq
import itertools

class Estado:
    def __init__(self, m_esq, c_esq, b_esq):
        self.m_esq = m_esq
        self.c_esq = c_esq
        self.b_esq = b_esq
    
    def valido(self):
        if self.m_esq < 0 or self.c_esq < 0 or self.m_esq > 3 or self.c_esq > 3:
            return False
        
        m_dir = 3 - self.m_esq
        c_dir = 3 - self.c_esq
        
        if self.m_esq > 0 and self.m_esq < self.c_esq:
            return False
        if m_dir > 0 and m_dir < c_dir:
            return False
            
        return True

    def eh_objetivo(self):
        return self.m_esq == 0 and self.c_esq == 0 and self.b_esq == 0

    def __eq__(self, outro):
        return (self.m_esq == outro.m_esq and 
                self.c_esq == outro.c_esq and 
                self.b_esq == outro.b_esq)

    def __hash__(self):
        return hash((self.m_esq, self.c_esq, self.b_esq))

    def __str__(self):
        margem = "esq" if self.b_esq == 1 else "dir"
        return f"(missionarios: {self.m_esq}, canibais: {self.c_esq}, barco: {margem})"


def heur(estado):
    return (estado.m_esq + estado.c_esq) / 2

def busc_a_star():
    estd_ini = Estado(3, 3, 1)
    
    fronteira = []
    count = itertools.count()
    heapq.heappush(fronteira, (heur(estd_ini), 0, next(count), estd_ini, [estd_ini]))
    
    visitados = set()
    
    movimentos_possiveis = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]

    while fronteira:
        f, g, _, estado_atual, caminho = heapq.heappop(fronteira)

        if estado_atual.eh_objetivo():
            return caminho

        if estado_atual in visitados:
            continue

        visitados.add(estado_atual)

        for m_mov, c_mov in movimentos_possiveis:
            if estado_atual.b_esq == 1:
                novo_estado = Estado(estado_atual.m_esq - m_mov, estado_atual.c_esq - c_mov, 0)
            else:
                novo_estado = Estado(estado_atual.m_esq + m_mov, estado_atual.c_esq + c_mov, 1)

            if novo_estado.valido() and novo_estado not in visitados:
                novo_g = g + 1
                novo_f = novo_g + heur(novo_estado)
                
                heapq.heappush(fronteira, (novo_f, novo_g, next(count), novo_estado, caminho + [novo_estado]))

    return None

if __name__ == "__main__":
    solucao = busc_a_star()

    if solucao:
        print("Solução encontrada em", len(solucao) - 1, "passos:")
        for i, passo in enumerate(solucao):
            print(f"{i}: {passo}")
    else:
        print("Nenhuma solução foi encontrada.")