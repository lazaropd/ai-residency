FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /modelchd

RUN mkdir /models
COPY pred.py .
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN echo "Dependencias instaladas com sucesso"

ENTRYPOINT ["python","pred.py"]

#RUN conda create -n docker_env -y
#SHELL ["conda","run","-n","docker_env","/bin/bash","-c"]
