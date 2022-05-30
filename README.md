# Project Structure :

* **.github/workflows**: CI/CD workflows
* **src** : contain all source code
  * **data**: script related to data (Scripts to download or generate data ...etc)
    * data_preparation.py 
  * **training**: script to train models and use it for prediction 
    * train.py : train the model
    * evaluate.py : evaluate the model
    * predict.py :  make prediction with the model
    * register.py : register the model
  * **util**: Python script for various utility operations specific to this ML project. **please stock each function in separate file**
    * name_function1.py 
    * name_function2.py
* **ml_pipeline:**  this folder for build ML pipeline. each folder for different ML pipeline
  * **plant_disease_pipeline**: this folder for stock pipeline files
    * dvc.yaml
    * training_dependencies.yml: conda dependencies for run this pipeline
    * params.yaml
* **dependencies** : 
  * ci_dependencies.yml : Conda dependencies for the CI environment.
  * mlflow_server_dependencies.yml : Conda dependencies for the mlflow server
  * local_env_dependencies.yml :  Conda  dependencies for local env
* **docs** :  markdown documentation for entire project.
* **notebooks** : Jupiter notebooks for experimentation. Naming convention is a number (for ordering),  the creator's initials, and a short `-` delimited description . example: `1.0-Hichem-initial-data-exploration`
* **API:** folder  for build APIs of this project
* **environment_setup**: everything related to infrastructure
  * mlflow_server_container:  the container that run MLflow tracking server.
    * Dockerfile
  * training_container: the container for build the training environment.
    * Dockerfile 
  * local_container: the container for build the local environment in developers and data scientists machines .
    * Dockerfile
  * Infra as code. yaml
* .**gitignore**: contain  [full_dataset, small dataset, results]. github will not contain dataset



Notes:

* the folder of the datasets is automaticly created in .dvc folder. so we delete the folder data from the structure
* results folder is not usesfull. we will separate the testing runs in separate group using mlflow  
*dsdsdsds
