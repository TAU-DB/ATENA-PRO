# ATENA-PRO
This repository contains the source code and experiments used to evalute ATENA-PRO, a framework for automatic discovery of personalized data stories, in the form of analytical
sessions. The repository is free for use for academic purposes. Upon using, please cite the paper:</br>
```Tavor Lipman, Tova Milo, and Amit Somech. ATENA-PRO: A Language and Architecture for Personalized Data Storytelling. PVLDB, 14(1): XXX-XXX, 2020```

## The problem: Auto-generating meaningful, coherent EDA sessions.
Deriving and communicating meaningful insights from a given dataset requires data analysts and scientists to develop complex data-driven “stories”. This is often done by thoroughly exploring and investigating the data for contextually-related patterns, gradually building a narrative, then communicating the findings in a compelling medium such as an infographic, report, or a scientific notebook.

## [Source Code](ATENA_PRO/src)
The source code is located [here](ATENA_PRO/src). <br/>
For installation guide, running instructions and further details please refer to the 
documentation under the source code directory in the link above.

## [Datasets](ATENA_PRO/datasets):
The datasets are located [here](ATENA_PRO/datasets). <br/>
ATENA-PRO is tested on 3 different datasets:
1. Netflix Movies and TV-Shows: lists about Netflix titles, each title is described using 11 features such as the country of production, duration/num. of seasons, etc.
2. Flight-delays: Each record describes a domestic US flight, using 12 attributes such as origin/destination airport, flight duration, issuing airline, departure delay times, delay reasons, etc.
3. Google Play Store Apps: A collection of mobile apps available on the Google Play Store. Each app is described using 11 features, such as name, category, price, num. of installs, reviews, etc.

## [DASL Queries](ATENA_PRO/queries):
The queries directory is located [here](ATENA_PRO/queries). <br/>
Specification of the DASL queries and tasks that were used in ATENA-PRO Experiments

## [User Study Notebooks](ATENA_PRO/user_study):
The user study notebooks are located [here](ATENA_PRO/user_study). <br/>
For assessing whether the generated analytical-sessions are indeed relevant to the analysis
tasks, and also interesting and coherent, and comparing ATENA-PRO to alternative approaches, we conducted a user study.
In the given link you can find the analytical-sessions that were presented to each participant of the user study.
The directory structure is as: `<Dataset>/<Task>/<Baseline>.ipynb` (the information of baseline wasn't given to the participants).
You can find the DASL query that matches each task number under [queries](ATENA_PRO/queries) section.
