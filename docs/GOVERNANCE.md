# Governance, workflow and reporting

This document keeps the project's working rules, moved here from the original README. It is addressed to the participating institutions.

## Data sources

For each participating country: firm-to-firm transaction data (VAT listings or e-invoicing) and firm level datasets (annual accounts, firm trade, location, sector, age).

## Data governance

**Data security protocols.** The project is a distributed micro data project. Security protocols are satisfied at the partner level and no confidential data is shared outside the partner's premises. All data protection regulations in place are documented by participating country.

**Reporting.** No individual data, nor data points that can be reverse engineered to recover individual firms or transactions, are reported. Distributions are reported such that at least 5 observations are within each cell (for example sector-year cells). Results are uploaded to a central, secure platform.

## Coding standards

**Language.** Python. Institutions using Stata 16 or newer can run the Python code within their Stata environment.

**Workflow.** The NBB collaborators write the code and send it to the partners, who run it on their datasets. Output consistent with the data security protocols is reported back to the NBB.

**Code templates.** Each country receives the same code to run on its own data. Coding etiquette is standardised and covers data cleaning, analysis and reporting scripts.

**Data harmonisation.** Common naming of variables and files, as set out in the main README ("Run it on your data").

**Version control.** GitHub, this repository.

**Documentation.** Guidelines for local officials to execute the code, interpret results and troubleshoot common issues are in the main README and in the task master files.

**Technology infrastructure.** A description of the computing environment is required to ensure local partners have the necessary resources to run the analyses. The pipeline has been run on the full Italian network (about 1.7 million firms and 59 million links) on a standard research server.

## Training and support

Dedicated training sessions familiarise local officials with the research design, code and reporting formats. The NBB team is available for technical issues, queries and clarifications.

## Pilot testing

Pilot tests on small samples (or on the synthetic data of task 0) identify potential issues in the code. Suggestions and improvements from the pilot phase are included before the full run.

## Full data analysis

The project is rolled out for all partners across all data.

## Reporting

After collecting and synthesising the results, the coordinating team presents initial results to the partners. Upon agreement, results are written into an article that each partner checks against its own data and reporting protocols before publication.
