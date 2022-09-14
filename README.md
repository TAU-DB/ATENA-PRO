# ANON-SYS
This repository contains the source code and experiments used to evaluate ANON-SYS, a framework for auto-generating personalized exploration notebook. 
The repository is free for use for academic purposes. Please contact the repository owners before usage.

## The problem: Auto-generating meaningful and relevant EDA sessions
One of the most effective methods for facilitating the process of exploring a dataset is to examine existing data exploration notebooks prepared by other data analysts or scientists. These notebooks contain curated sessions of contextually-related query operations that all together demonstrate interesting hypotheses and conjectures on the data. Unfortunately, relevant such notebooks, that had been prepared on the same dataset, and in light of the
same analysis task – are often nonexistent or unavailable. 

## [Source Code](ANON_SYS/src)
The source code is located [here](ANON_SYS/src) <br/>
For installation guide, running instructions and further details please refer to the 
documentation under the source code directory in the link above.

## [Technical Report](ANON_SYS/technical_report)
The paper's technical report is located [here](ANON_SYS/technical_report/Technical&#32;Report.pdf). <br/>

## [Datasets](ANON_SYS/datasets)
The datasets are located [here](ANON_SYS/datasets). <br/>
ANON-SYS is tested on 3 different datasets:
1. Netflix Movies and TV-Shows: list of Netflix titles, each title is described using 11 features such as the country of production, duration/num. of seasons, etc.
2. Flight-delays: Each record describes a domestic US flight, using 12 attributes such as origin/destination airport, flight duration, issuing airline, departure delay times, delay reasons, etc.
3. Google Play Store Apps: A collection of mobile apps available on the Google Play Store. Each app is described using 11 features, such as name, category, price, num. of installs, reviews, etc.

## [LDX Queries](ANON_SYS/queries)
The queries directory is located [here](ANON_SYS/queries). <br/>
Specification of the LDX queries and tasks that were used in ANON-SYS Experiments.

## [User Study Notebooks](ANON_SYS/user_study)
The user study notebooks are located [here](ANON_SYS/user_study). <br/>
For assessing whether the generated exploratory sessions are indeed relevant to the analysis tasks, and also interesting and coherent,
and to compare ANON-SYS to alternative approaches, we conducted a user study.
In the given link you can find the exploratory sessions that were presented to each participant of the user study.
The directory structure is as: `<Dataset>/<Task>/<Baseline>.ipynb` (the identity of the baseline wasn't given to the participants).
You can find the LDX query that matches each task number under [queries](ANON_SYS/queries) section.
