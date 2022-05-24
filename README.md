# Project Structure :

* **.github/workflows**: CI/CD workflows
<<<<<<< HEAD
* **data:**
  * **prototype_dataset:** small dataset for quick train and evaluate model; doesn't include all data(used in local development)
  * **full_dataset:** the complete dataset (used in training env in cloud)
=======
>>>>>>> e7de4e4457dd024f1e8bc746b2a249bf0f4c4554
* **src** : contain all source code
  * **data**: script related to data (Scripts to download or generate data ...etc)
    * data_preparation.py 
  * **training**: script to train models and use it for prediction 
    * train.py : train the model
    * evaluate.py : evaluate the model
    * predict.py :  make prediction with the model
    * register.py : register the model
<<<<<<< HEAD
    * serving.py : build api of model
  * **pipeline:** load script of each step and build pipeline
    * training_pipeline.py: load script of each step and build the pipeline in the training env in cloud
  * **util**: Python script for various utility operations specific to this ML project
  * **parameters.json**: to stock paramater of training, the script of training load the parameter from here 
* **dependencies** : 

  * training_dependencies.yml: conda dependencies for training
  * ci_dependencies.yml: Conda dependencies for the CI environment.
* **docs** :  markdown documentation for entire project.
* **notebooks** : Jupiter notebooks for experimentation. Naming convention is a number (for ordering),  the creator's initials, and a short `-` delimited description . example: `1.0-Hichem-initial-data-exploration`
* **environment_setup**: everything related to infrastructure
  * Dockerfile
  * Infra as code yaml
* .**gitignore**
* **README.md**       <- The top-level README for developers using this project.
=======
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
>>>>>>> e7de4e4457dd024f1e8bc746b2a249bf0f4c4554
