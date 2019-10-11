#!/home/lazarop/miniconda3/envs/python/bin/python

import string

def filtraletra(residentes, letra):
    residentes = list(map(lambda x: x.lower(), residentes))
    residentes.sort()
    if type(letra) == str:
        letra = letra.lower()
    else:
        return "Favor informar uma e apenas uma letra!"
    lista_filtrada = []
    if 0 < len(letra) <= 1:
        lista_filtrada = [residente for residente in residentes if residente[0] != letra]
    else:
        print("Favor informar uma e apenas uma letra!")
    return lista_filtrada
