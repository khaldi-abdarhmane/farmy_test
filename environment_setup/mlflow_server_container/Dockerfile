# docker mlflow server 

FROM continuumio/miniconda3

RUN conda install -c conda-forge mlflow && conda install boto3 &&  conda install -c anaconda psycopg2

CMD mlflow ui --backend-store-uri $DB_URI  --default-artifact-root $S3_URI --host 0.0.0.0 


