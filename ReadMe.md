# Disaster Response Pipeline

## Description
This project covers the data engineering skills, I applied this skills to analyze disaster data from [https://www.figure-eight.com/](Figure Eight) and built a model and a website that's using that model to classify the new messages. This project is part of Udacity Data Scientist Nanodegree.

## Content
- data/
	- categories.csv
	- messages.csv
	- etl_pipeline.py 	=> the script to extract and clean data
- models/
	- train_classifier.py  => script to regenerate the classification model
- app/
	- run.py  => Flask web applications
	- templates/
		- go.html
		- master.html
- ETL Pipeline Preparation.ipynb
- ML Pipeline Preparation.ipynb
- ReadMe.md

## How to Run
### ETL Model (data folder)
write the following command in the CMD
```
python etl_pipeline.py messages.csv categories.csv DisasterResponse.db
```
### Classification Model (model folder)
write the following command in the CMD
```
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```

### The Web App. (app folder)
write the following command in the CMD
```
python run.py
```
