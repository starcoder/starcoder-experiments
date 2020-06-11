# StarCoder experiments

This repository showcases various ways that [StarCoder](https://github.com/TomLippincott/starcoder) can be used to model relational data.  It includes a number of data sets, described below, with associated schemas describing the entities, fields, and relationships therein.  A build system is used to marshal the data, train models, and examine the output.  To get started quickly, after cloning this repository, invoke the following commands to set up the environment:

```
cd starcoder-experiments
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp custom.py.example custom.py
```

You can then invoke `scons` to run the build system, which will perform the simplest experiment (solving random arithmetic expressions).  By editing the `custom.py` file, you can uncomment other experiments and set/alter variables for e.g. data locations and parameters.

## Data sets

| Name                         | Source description                                                                                                                                                      | Entities                                         |
| ---                          | ---                                                                                                                                                                     | ---                                              |
| arithmetic                   | Randomly-generated expression trees where the leaves are constants and the internal nodes are unary or binary operations                                                | nodes                                            |
| reddit                       | Comment trees from Reddit                                                                                                                                               | comments, users, subreddits                      |
| fluency                      | "Snowflakes" of multi-lingual Twitter users (shallow follower network and up to 200 tweets per user), annotated with language model scores                              | tweets, users                                    |
| linguistic\_lid              | Twitter's language ID data set with languages linked according to the WALS hierarchy                                                                                    | tweets, languages, genuses, families, macroareas |
| targeted\_sentiment          | The Stanford Sentiment Treebank                                                                                                                                         | nodes                                            |
| maroon\_ads                  | Notices of "maroons", escaped slaves in the Caribbean islands                                                                                                           | slaves, notices, cities, owners                  |
| affiches\_americaines        | Price information for a variety of common products over time, as recorded in several Caribbean cities around the time of the French Revolution                          | products, cities, listings                       |
| royal\_inscriptions          | Translations of Cuneiform inscriptions from the Ancient Near East, linked according to the ruler they were made by/for, and thereby to dynasties and historical periods | texts, witnesses, rulers, dynasties, periods     |
| middle\_english              | Chaucer's Canterbury Tales, with metrical and grammatical information attached to each word                                                                             | books, sections, lines, words                    |
| entertaining\_america        | Records of late-19th-century traveling performances throughout America                                                                                                  | performances, performers, locations, notices     |
| post\_atlantic\_slave\_trade | Manifests and escape notices from the Baltimore area in the time period following the ban of the international slave trade                                              | voyages, slaves, vessels, notices, gazettes      |
| paris\_tax\_rolls            | Tax collection records from Medieval Paris                                                                                                                              | parishes, streets, people, occupations, debts    |
| documentary\_hypothesis      | The oldest extant manuscript of the Hebrew bible, annotated according to the sources in the Documentary Hypothesis                                                      | sources, documents, books, chapters, verses      |
| women\_writers               | Collection of novels, poetry, and correspondence written by women before the modern era                                                                                 | authors, books, sections                         |

