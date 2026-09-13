# tasks/raw_data

Put your two input tables here (see the main README, "Run it on your data"). Real micro data must never be committed: the `.gitignore` excludes everything in this folder except the files listed below.

Files shipped with the repository, used by the monetary policy counterfactual in `task3_network_statistics/src/monpol.py`:

| File | Content |
|---|---|
| `irfs_int_avg_up_quarterly.csv` | Belgian local projection coefficients for real sales by horizon: baseline effect of the monetary policy shock, interaction with (moving average) upstreamness, and controls. Estimated in Dhyne, Magerman and Palazzolo (2026). |
| `irfs_p_int_avg_up_quarterly.csv` | Same for producer prices. |
| `vcov_int_avg_up_quarterly.csv`, `vcov_p_int_avg_up_quarterly.csv` | Variance-covariance matrices of the coefficients above. |
| `gdp_data.csv` | World Bank GDP and GDP per capita in current USD, 2002 to 2024, for the five countries. |

These are aggregate estimates, not micro data.
