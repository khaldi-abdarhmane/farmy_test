# Project Structure :

* **.github/workflows**: CI/CD workflows
* **data:**
  * **prototype_dataset:** small dataset for quick train and evaluate model; doesn't include all data(used in local development)
  * **full_dataset:** the complete dataset (used in training env in cloud)
* **src** : contain all source code
  * **data**: script related to data (Scripts to download or generate data ...etc)
    * data_preparation.py 
  * **training**: script to train models and use it for prediction 
    * train.py : train the model
    * evaluate.py : evaluate the model
    * predict.py :  make prediction with the model
    * register.py : register the model
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
