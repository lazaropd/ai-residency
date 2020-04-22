

import os
import numpy as np

import multiprocessing
from multiprocessing import Process, Pool, current_process

from threading import Thread
from queue import Queue

import time

SIZE = 50 # número de repetições a ser realizada
BUFFER = 20


# a função que será paralelizada para outros núcleos
def myIntensiveTask(size):
    X = np.random.rand(size, size)
    Y = np.random.rand(size, size)
    M = np.matmul(X, Y)
    proc_name = current_process().name
    #print('Matriz de tamanho {0} processada pelo: {1}'.format(size, proc_name))
    return M.shape


# vamos colocar cada processo a ser executado em uma lista
def myPlanner(numbers):
    procs = []
    for index, number in enumerate(numbers):
        proc = Process(target=myIntensiveTask, args=(number,))
        procs.append(proc)
    return procs


# inicializa e executa as atividades planejadas
def runProcs(procs):
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()


# vamos colocar todos os processos em um pool e designar workers para trabalharem
def myCallBack(result):
    global results
    results.append(result)

def myWorkersPool(numbers, workers):
    pool = multiprocessing.Pool(workers)
    for i, number in enumerate(numbers):
        pool.apply_async(myIntensiveTask, args=(number,), callback=myCallBack)
    pool.close()
    pool.join()


# aqui simulamos um processo lento que será empurrado para outra thread
def myLaggyFunction():
    global SIZE
    global queue1
    stop = False
    i = 0
    while True:
        if stop: return
        if not queue1.full():
            number = 100 * np.random.randint(10, 20)
            queue1.put(number)
            i += 1
        if i > SIZE: stop = True
        time.sleep(0.1) # simulando uma demora de 100ms em cada run

# esta função faz a mesma coisa que a anterior, mas sem usar queues
def myLaggyRaw():
    global SIZE
    numbers = []
    stop = False
    i = 0
    while True:
        if stop: return numbers
        number = 100 * np.random.randint(10, 20)
        numbers.append(number)
        time.sleep(0.1)
        i += 1
        if i > SIZE: stop = True
    return numbers

# aqui recuperamos o próximo elemento pronto e disponível na lista
def getQueued(i):
    global queue1
    if queue1.qsize() > 0:
        number = queue1.get()
        #print('Processando:', i+1, 'para matriz de tamanho', number, 'tendo ainda', queue1.qsize(), 'números na fila!')
        myIntensiveTask(number)
        return True
    return False

# uma função genérica para inicializar uma tarefa em outro núcleo
def startMyTask(task, delay=0):
    t = Thread(target=task, args=())
    t.daemon = True
    t.start()
    time.sleep(delay)



# vamos organizar uma lista de chamadas para a função
# vamos randomizar o tamanho para forçar inversões na ordem de entregas dos jobs
numbers = [100 * np.random.randint(10, 20) for i in range(SIZE)]

# vamos primeiro rodar sem multiprocessamento e usar apenas 1 núcleo para comparar
os.environ["OMP_NUM_THREADS"] = '1'
start = time.time()
for index, number in enumerate(numbers):
    myIntensiveTask(number)
print('Tempo de execução usando apenas 1 núcleo: %.2f' % (time.time() - start))
time.sleep(1)


# para rodar todos os processos paralelamente e usando todos os recursos
procs = myPlanner(numbers)
start = time.time()
runProcs(procs)
print('Tempo de execução iniciando todos os processos simultaneamente: %.2f' % (time.time() - start))
time.sleep(1)


# vamos finalmente comparar o desempenho usando diferentes números de workers
for workers in range(1, int(0.9 * multiprocessing.cpu_count())):
    results = []
    start = time.time()
    myWorkersPool(numbers, workers)
    print('Tempo de execução usando %d workers no POOL: %.2f' % (workers, time.time() - start))
    time.sleep(1)


# esse gerenciamento pode ser feito de várias formas, usando queues por exemplo
# abaixo uma implementação simulando um processo lento sendo removido da thread principal
start = time.time()
queue1 = Queue(maxsize=BUFFER)
startMyTask(myLaggyFunction, 1)
i = 0
while i < SIZE:
    if getQueued(i): i += 1
print('Tempo de execução colocando a função lagger em paralelo: %.2f' % (time.time() - start))

# finalmente rodando a função lagger na thread principal
start = time.time()
numbers = myLaggyRaw()
i = 0
for number in numbers:
    myIntensiveTask(number)
print('Tempo de execução mantendo a função lagger na thread principal: %.2f' % (time.time() - start))



# export OMP_NUM_THREADS=1 ; time python aula.py
# para checar a utilização de núcleos usar top, em vez de htop
