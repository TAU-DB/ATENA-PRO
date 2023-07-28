# ANON-SYS
This repository contains the source code and experiments used to evaluate ANON-SYS, a framework for auto-generating personalized exploration notebook. 
The repository is free for use for academic purposes. Please contact the repository owners before usage.

## The problem: Auto-generating meaningful and relevant Exploratory Notebooks
One of the most effective methods for facilitating the process of exploring a dataset is to examine existing data exploration notebooks prepared by other data analysts or scientists. These notebooks contain curated sessions of contextually-related query operations that all together demonstrate interesting hypotheses and conjectures on the data. Unfortunately, relevant such notebooks, that had been prepared on the same dataset, and in light of the
same analysis task – are often nonexistent or unavailable. ANON-SYS is a CDRL framework for autogenerating interesting, task-relevant notebooks given a user-provided dataset and specifications.  

## [Source Code](ANON_SYS/src)
The source code is located [here](ANON_SYS/src) (ANON_SYS/src) <br/>
For installation guide, running instructions and further details please refer to the 
documentation under the source code directory in the link above.

## [LDX Technical User Guide](ANON_SYS/LDX_user_guide.pdf)
LDX user technical guide is located [here](ANON_SYS/LDX_user_guide.pdf). <br/>

## [Experiment Datasets](ANON_SYS/datasets)
The datasets used in our empirical evaluation are located [here](ANON_SYS/datasets) (ANON_SYS/datasets). <br/>
ANON-SYS is tested on 3 different datasets:
1. Netflix Movies and TV-Shows: list of Netflix titles, each title is described using 11 features such as the country of production, duration/num. of seasons, etc.
2. Flight-delays: Each record describes a domestic US flight, using 12 attributes such as origin/destination airport, flight duration, issuing airline, departure delay times, delay reasons, etc.
3. Google Play Store Apps: A collection of mobile apps available on the Google Play Store. Each app is described using 11 features, such as name, category, price, num. of installs, reviews, etc.

## [LDX Queries](ANON_SYS/queries)
The LDX queries directory is located [here](ANON_SYS/queries) (ANON_SYS/queries). <br/>
Specification of the LDX queries and tasks that were used in ANON-SYS Experiments. <br/>
For an overview and explanations for each of the LDX Specifications -- see [this document](ANON_SYS/queries/Queries%20Overview.pdf).

## [User Study Notebooks](ANON_SYS/user_study)
The exploration notebooks generated by either ANON-SYS and the baselines are located [here](ANON_SYS/user_study) (ANON_SYS/user_study). <br/>
In the given link you can find the exploratory sessions that were presented to each participant of the user study.
The directory structure is as: `<Dataset>/<Task>/<Baseline>.ipynb` (the identity of the baseline wasn't given to the participants).
For the ChatGPT-based notebooks, we also provide the prompt and raw output. 



