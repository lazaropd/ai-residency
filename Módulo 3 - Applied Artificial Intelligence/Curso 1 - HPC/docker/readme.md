# Docker para treino e predição usando a GPU

* o uso de Dockers elimina a necessidade de VMs, entretanto, cada Docker consome considerável armazenamento
* um Dockerfile com os requirements, fontes e dados para train
* um Dockerfile com os requirements, fontes e dados para predictions

# Etapas (repetir para cada Docker)

* criar pasta para organizar os arquivos
* adicionar todos os códigos à pasta
* criar o requirements.txt com os packages necessários
* criar o Dockerfile com todas as instruções:
    * imagem do SO, eventualmente com instalações prévias (ver DockerHub)
        * para tf-gpu com py3, usamos tensorflow/tensorflow:latest-gpu-py3
        * pesquisar outras opções aqui: https://hub.docker.com/r/tensorflow/tensorflow/tags/?page=1&name=latest-
    * copiar os arquivos necessários para o contêiner
    * instalar/atualizar softwares necessários
    * criar e ativar o ambiente se necessário
    * preparar o entry point
* no terminal, navegar até o diretório do Dockerfile
* criar e testar a imagem para train:
    * sudo docker build -t lazaropd/chdgpu:train .
    * sudo docker run -v /home/lazarop/dockerfiles:/models -it --rm --gpus all lazaropd/chdgpu:train
        * esta instrução cria um novo volume e o caminho models/ para compartilhar conteúdos entre o Docker e a máquina local
* navegar para o diretório do Dockerfile de predictions
* criar e testar a imagem para prediction:
    * sudo docker build -t lazaropd/chdgpu:pred .
    * sudo docker run -v /home/lazarop/dockerfiles:/modelchd/models -it --rm --gpus all lazaropd/chdgpu:pred
  
# Lista de comandos úteis

* **Criar a imagem:** sudo docker build --tag myapp .

* **Rodar a imagem:** sudo docker run --rm -d -p 8501:8501 -it --gpus all myapp

* **Listar dockers em execução:** sudo docker ps -a

* **Listar imagens criadas:** sudo docker images

* **Encerrar o docker em execução:** sudo docker rm --force myapp

* **Conectar ao docker hub:** sudo docker login

* **Atribuir uma tag para o usuário e versão:** sudo docker mysource:tag mytarget:tag

* **Subir imagem para o docker hub:** sudo docker image myuser/myapp:version

* **Para parar todos os contêineres:** sudo docker kill $(sudo docker ps -q)

* **Caso precise remover uma imagem:** sudo docker image rm myapp [OR ID]

* **Para remover todos os dockers, imagens e volumes:**
    * sudo docker rm -vf $(sudo docker ps -a -q)
    * sudo docker rmi -f $(sudo docker images -a -q)
    * sudo docker system prune -a --volumes



