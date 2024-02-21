# ECB multi-country B2B analysis 
coordinated by Emmanuel Dhyne, Glenn Magerman, and Alberto Palazzolo

For any question, feel free to reach out to [Glenn Magerman](glenn.magerman@ulb.be) or [Emmanuel Dhyne](emmanuel.dhyne@nbb.be).

## 1. Overview
The goal of this project is to develop key statistics on firm-to-firm networks across multiple EU countries in the context of the multi-year [Research Network ChaMP](https://www.ecb.europa.eu/pub/economic-research/research-networks/html/champ.en.html) of the European Central Bank.
This GitHub repository is aimed at the collaborators of the project, and contains all codes to construct the various statistics for the project, from data pre-cleaning, cleaning, harmonization, to descriptive statistics.

Section 2 describes the data setup, governance and project workflow. 

Section 3 describes the code structure to produced the targeted network statistics. (TBD).

## 2. Data
### 2.1 Data sources
For each participating country, firm-to-firm data, firm-level datasets (annual accounts, firm trade, location/sector/age etc.).

### 2.2	Data governance
**Data security protocols:** The project is a distributed micro-data project. This guarantees that security protocols are satisfied at the partner level, and that no confidential data is shared outside the partner’s premises. All data protection regulations that are in place should be documented by participating country.

**Reporting:** No individual data, nor data points that can be reverse engineered to recover individual firms or transactions will be reported. Reporting of distributions is such that at least 5 observations are within each cell (e.g. sector-year observations). Results will be uploaded to a central, secure platform.

### 2.3 Coding standards
**Language:** Coding will be in Python. NCBs which are using Stata 16 or newer will be able to run the python code within their Stata environment.

**Workflow:** NBB collaborators will write the codes, and send these to the individual partners to be run on their datasets. Output, consistent with data security protocols, will be reported back to the NBB. 

**Code templates:** Each country will receive the same code to be run on their own data. Coding etiquette is standardized, and includes codes for data cleaning, analysis, and reporting scripts. 

**Data harmonization:** Collaboration rules to be implemented (e.g. naming variables, files, etc.)

**Version control:** We use GitHub for version control of codes. 

**Documentation:** We will provide thorough documentation and guidelines for the local officials to execute their code, interpret results, and troubleshoot common issues.

**Technology infrastructure:** A description of the computing environment is required to ensure local partners have the necessary computing resources to run the analyses. 

### 2.4. Training and support
We will set up dedicated training sessions for the local officials to familiarize them with the research design, code, and reporting formats. The NBB team will be available for assistance with technical issues, queries, and clarifications.

### 2.5.	Pilot testing
We will run pilot tests with small samples to identify potential issues in the codes. At this stage of the process, suggestions and improvements from the pilot phase will be included.

### 2.6.	Full data analysis
The project is rolled out for all partners across all data.

### 2.7	Reporting
After collecting and synthesizing the results, the NBB partners will present initial results for reporting, to be discussed with the partners. Upon agreement, results are written into an article, that can be checked against data and reporting protocols of each partner, for publication as ECB WP.

## 3. Code structure

### 3.1 General overview
The project is constructed as modular tasks. Each task has an input (data or previous results), a function (the code), and an output (the result of the code).
The output of one task can be the input for another task. The entirety of the code structure is a topological sort, with a downstream flow. An output of a task cannot be used as the input of a previous task. Tasks are short, clear and well-defined. Each task also contains a master file, that (i) initializes the task ("erase previous results"), (ii) runs the codes, (iii) creates output. This ensures that there are no previous results that are inconsistent with the current state of the code, which might otherwise percolate throughout the project.

There is one main code script to rule them all. This main script initializes the project. Here you can set the proper folder location to execute all the codes. It also contains all macros that are common to the project (e.g. the time coverage of the data), as well as layout choices for graphs etc. Finally, this script calls the master scripts for each task, which in turn call all the codes within that task.

This structure ensures full reproducibility, minimizes errors, and makes codes very easy to follow. If something needs to be changed, it is clear where one has to look.
The entire code pipeline can then be submitted to e.g. a journal for its replication packages.


