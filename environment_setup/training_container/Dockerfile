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
RUN conda deactivate
RUN conda activate myenv
RUN conda install boto3

CMD [ "git clone --single-branch --branch khaldi_refactor_class_to_one_file https://github.com/khaldi-abdarhmane/farmy_test.git" ,"dvc pull data"]