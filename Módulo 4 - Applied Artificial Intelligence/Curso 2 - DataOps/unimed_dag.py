# Importando as bibliotecas que vamos usar nesse exemplo
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python_operator import PythonOperator

from unimed.functions import *


# Definindo alguns argumentos básicos
default_args = {
    'owner': 'lazaropd',
    'depends_on_past': False,
    'email': ['aiflow@email.com'],
    'email_on_failure': False,
	'email_on_retry': False,
    'start_date': datetime(2020, 7, 1),
    'retries': 1,
}

# Nomeando a DAG e definindo quando ela vai ser executada (você pode usar argumentos em Crontab também caso queira que a DAG execute por exemplo todos os dias as 8 da manhã)
with DAG(
    'unimed_prontuarios',
    description = 'ETL - extração de documento pdf para database estruturado CSV',
    schedule_interval = timedelta(minutes=1),
    catchup = False,
    default_args = default_args
    ) as dag:

    # Definindo as tarefas que a DAG vai executar, nesse caso a execução de dois programas Python, chamando sua execução por comandos bash
    pdf_to_images = PythonOperator(
        task_id = 'pdf_to_images',
        python_callable = pdfToImages)

    adjust_rotation = PythonOperator(
        task_id = 'adjust_rotation',
        python_callable = adjustRotation)

    read_by_ocr = PythonOperator(
        task_id = 'read_by_ocr',
        python_callable = readByOCR)

    save_database = PythonOperator(
        task_id = 'save_database',
        python_callable = saveDatabase)

    # Definindo o padrão de execução
    pdf_to_images >> adjust_rotation >> read_by_ocr >> save_database