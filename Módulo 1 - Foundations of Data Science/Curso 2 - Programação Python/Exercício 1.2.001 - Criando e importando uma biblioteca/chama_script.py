#!/home/lazarop/miniconda3/envs/python/bin/python

import sys
#sys.path.append('libraries_path')

from script import filtraletra

residentes = ["André", "Daniel", "Cristiano", "Wana", "Duan", "Harrison", "Alessandra", "Bruna", "Alexandre", "Lázaro", "Vitor", "Leonardo", "Sávio", "Lucas", "Rafael", "Jéssica", "Letícia", "Gabriel", "Gabriela", "Felipe", "Muriel"] 

print(filtraletra(residentes, "ad"))
print(filtraletra(residentes, "A"))
print(filtraletra(residentes, 5))
