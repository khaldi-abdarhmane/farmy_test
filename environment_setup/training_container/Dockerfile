FROM continuumio/miniconda3:latest
RUN mkdir -p /scripts
COPY workstation.sh /scripts
WORKDIR /scripts
RUN chmod +x workstation.sh
RUN ./workstation.sh
WORKDIR /usr/src/app
COPY ml_pipelines/plant_disease_pipeline/requirements.txt .
RUN conda create -n myenv
RUN conda activate myenv
RUN conda install --file requirements.txt
RUN rm -rf requirements.txt
COPY . .
