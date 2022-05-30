FROM continuumio/miniconda3:latest
RUN mkdir -p /scripts
COPY workstation.sh /scripts
WORKDIR /scripts
RUN chmod +x workstation.sh
RUN ./workstation.sh
WORKDIR /usr/src/app
COPY requirements.txt .
RUN conda create -n myenv
# RUN conda activate myenv
RUN conda install --file requirements.txt
RUN conda install -c conda-forge mlflow
RUN rm -rf requirements.txt
# COPY . .