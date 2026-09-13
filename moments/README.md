# Key moments

`key_moments.csv` collects 43 harmonised firm-to-firm network statistics for Belgium (BE), Estonia (EE), Hungary (HU), Italy (IT) and Portugal (PT), base year 2019, taken from the tables of Magerman et al. (2026), "Beyond firm size". Every row names the paper table it comes from.

Columns: `moment` (short identifier), `description`, `unit` (count, share, correlation, exponent or ratio), one column per country, `source` (paper table).

Groups of rows:

- **Coverage** (Table 1): firms, sellers, buyers, relationships. Sample is the giant connected component of firms with positive sales and at least one domestic B2B transaction.
- **Density and aggregation** (Table 6): network density at the firm level and for bottom-up sector networks at NACE 2 and 4 digits, and the within-sector (diagonal) share of intermediate inputs at the sector level.
- **Concentration** (Table 2): share of network sales of the top 10% and 1% of firms, and of bilateral sales value of the top 10% and 1% of supplier-customer relationships ranked by annual value.
- **Degree distributions** (Tables B.2 and B.3): mean and percentiles of the number of customers and the number of suppliers per firm.
- **Tail exponents** (Table B.1): Hill estimates with the Clauset et al. (2009) cutoff for total sales, network sales, number of customers and number of suppliers.
- **Correlations** (Table 3): Pearson correlations of log variables, demeaned within NACE 2-digit industries.
- **Firm size variance decomposition** (Table 4): exact decomposition of log sales into seller fixed effect, number of customers, average customer fixed effect, covariance term and sales outside the network, following Bernard et al. (2022). Shares sum to one.
- **Input share dispersion** (Table 7): percentiles across supplier-customer sector pairs of the coefficient of variation of firm level input shares (m_ij / total inputs of j) within the pair.
- **Monetary policy counterfactual** (Table 9): share of the aggregate sales and price response to a common ECB shock carried by firms with upstreamness above 4, applying Belgian estimates to each country's upstreamness distribution.

All statistics are aggregate, satisfy each institution's confidentiality protocol (at least 5 observations per reported cell) and are comparable across countries. They reproduce results from the paper and remain subject to the relevant institutional publication and data governance rules. Please cite the paper when using them.
