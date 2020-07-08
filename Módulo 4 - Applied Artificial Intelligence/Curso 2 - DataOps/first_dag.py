# Importando as bibliotecas que vamos usar nesse exemplo
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.bash_operator import BashOperator

# Definindo alguns argumentos básicos
default_args = {
   'owner': 'lazaropd',
   'depends_on_past': False,
   'start_date': datetime(2020, 7, 1),
   'retries': 0,
}

# Nomeando a DAG e definindo quando ela vai ser executada (você pode usar argumentos em Crontab também caso queira que a DAG execute por exemplo todos os dias as 8 da manhã)
with DAG(
   'first_dag',
   schedule_interval=timedelta(minutes=1),
   catchup=False,
   default_args=default_args
   ) as dag:

    # Definindo as tarefas que a DAG vai executar, nesse caso a execução de dois programas Python, chamando sua execução por comandos bash
    t1 = BashOperator(
    task_id='first_script',
    bash_command="""
    cd /mnt/c/dags/first_project/
    python3 first_script.py
    """)

    t2 = BashOperator(
    task_id='second_script',
    bash_command="""
    cd /mnt/c/dags/first_project/
    python3 second_script.py
    """)

    # Definindo o padrão de execução, nesse caso executamos t1 e depois t2
    t1 >> t2