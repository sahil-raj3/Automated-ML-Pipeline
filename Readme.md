
# Automated ML Pipeline

## What is AutoML?
Automated Machine Learning provides methods and processes to make Machine Learning available for non-Machine Learning experts, to improve efficiency of Machine Learning and to accelerate research on Machine Learning.

Machine learning (ML) has achieved considerable successes in recent years and an ever-growing number of disciplines rely on it. However, this success crucially relies on human machine learning experts to perform the following tasks:

1. Preprocess and clean the data.
2.Select and construct appropriate features.
3.Select an appropriate model family.
4.Optimize model hyperparameters.
5.Postprocess machine learning models.
6.Critically analyze the results obtained.

As the complexity of these tasks is often beyond non-ML-experts, the rapid growth of machine learning applications has created a demand for off-the-shelf machine learning methods that can be used easily and without expert knowledge. We call the resulting research area that targets progressive automation of machine learning AutoML.

_To know more check out official website of [Automl](https://www.automl.org/automl/)._

## About Project
our objective is to made a platform where user can provide dataset or a website url to scrape data directly without any downloading process and gets results with best model. Where all the background tasks (like data splitting model selection, model training, model comparison and hyperparameter optimization) are performed by automated pipeline.

In this Project we used [EvalML](https://evalml.alteryx.com/en/v0.11.2/start.html) an ```AutoML``` library that builds, optimizes, and evaluates machine learning pipelines using domain-specific objective functions.

Streamlit library is used to design the user interface for this project.


## How to Run?
First of all clone or download this code into your local. ```Don`t know how? Just checkout the main branch readme file.```

After this We need an environment to run our project. Fire up your terminal and create a python virtual environment by following command:

```pip -m venv <name of environment>```

Once you have created your environment you need to add dependencies required for this project.

Install them by running command: ```pip install -r requirements.txt```

Done with this now run you project by : ```streamlit run automated_pipeline.py```

**********
# Thankyou
