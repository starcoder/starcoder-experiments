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

| Name                         | Description | Entities                                      |
| ---                          | ---         | ---                                           |
| arithmetic                   |             | constants, operations                         |
| reddit                       |             | comments, users, subreddits                   |
| fluency                      |             | tweets, users                                 |
| linguistic\_lid              |             | tweets, languages, genuses                    |
| targeted\_sentiment          |             | parse nodes                                   |
| maroon\_ads                  |             | slaves, notices, cities, owners               |
| affiches\_americaines        |             | products, cities, listings                    |
| royal\_inscriptions          |             | texts, witnesses, rulers, dynasties, periods  |
| middle\_english              |             | books, sections, lines, words                 |
| entertaining\_america        |             | performances, performers, locations, notices  |
| paris\_tax\_rolls            |             | parishes, streets, people, occupations, debts |
| post\_atlantic\_slave\_trade |             | voyages, slaves, vessels, notices, gazettes   |
| documentary\_hypothesis      |             | sources, documents, books, chapters, verses   |
| women\_writers               |             | authors, books, sections                      |

