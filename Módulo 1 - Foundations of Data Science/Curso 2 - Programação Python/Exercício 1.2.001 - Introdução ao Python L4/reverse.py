#!/home/lazarop/miniconda3/envs/python/bin/python

import sys

file_name = sys.argv[1]

def open_file(file):
    with open(file, "r") as file:
        content = file.readlines()
        file.close()
        print("Arquivo carregado com sucesso\n")
        return content
        
def reverse_text(linhas):
    reverso = linhas[::-1]
    texto = "\n".join(item.replace("\n", "") for item in reverso)
    return texto

if __name__ == "__main__":
    print(reverse_text(open_file(file_name)))
