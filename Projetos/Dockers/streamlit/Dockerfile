FROM continuumio/miniconda3
COPY . /app
WORKDIR /app
RUN conda create -n docker_env -y
SHELL ["conda","run","-n","docker_env","/bin/bash","-c"]
RUN echo "Ambiente ativado com sucesso"
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit","run","demo.py"]