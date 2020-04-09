
import argparse

from myclass import my_operations


parser = argparse.ArgumentParser(description='Exemplo com parser')
parser.add_argument('number', metavar='Número', type=int, 
                    help='Informar um número inteiro')
parser.add_argument('-o', metavar='--operacao', default='linear', type=str, dest='oper',
                    help='Informar a operação (linear ou quadratic)')

args = parser.parse_args()
number = args.number
operation = args.oper

obj = my_operations(number)

print('Mostrando progressão para o número:', number)
if operation == 'linear':
    print(obj.linear())
elif operation == 'quadratic':
    print(obj.quadratic())
else:
    print('Operação inválida!')
    parser.print_usage()
