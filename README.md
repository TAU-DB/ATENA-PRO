# ATENA-PRO Benchamark
This repository contains the automatic benchmark used to evalute ATENA-PRO,  a framework for automatic discovery of personalized data stories, in the form of analytical
sessions. The repository is free for use for academic purposes. Upon using, please cite the paper:</br>
```Tavor Lipman, Tova Milo, and Amit Somech. ATENA-PRO: A Language and Architecture for Personalized Data Storytelling. PVLDB, 14(1): XXX-XXX, 2020```

## The problem: Auto-generating meaningful, coherent EDA sessions.
Deriving and communicating meaningful insights from a given dataset requires data analysts and scientists to develop complex data-driven “stories”. This is often done by thoroughly exploring and investigating the data for contextually-related patterns, gradually building a narrative, then communicating the findings in a compelling medium such as an infographic, report, or a scientific notebook.

## The code - [ATENA-PRO Benchmark](benchmark/simulation)
ATENA-PRO benchmark as used in our experiments described in the paper
The benchmark, as used in our experiments, includes 3 datasets: lights, Netflix content and Google play store.
We additionally provide implementations for all of our evaluation metrics. 

## The [datasets](benchmark/datasets):
ATENA-PRO is tested on 3 different datasets:
1. Netflix Movies and TV-Shows: lists about Netflix titles, each title is described using 11 features such as the country of production, duration/num. of seasons, etc.
2. Flight-delays: Each record describes a domestic US flight, using 12 attributes such as origin/destination airport, flight duration, issuing airline, departure delay times, delay reasons, etc.
3. Google Play Store Apps: A collection of mobile apps available on the Google Play Store. Each app is described using 11 features, such as name, category, price, num. of installs, reviews, etc.

## The [queries](benchmark/queries):
Specification of the DASL queries and tasks that were used in ATENA-PRO Experiments
