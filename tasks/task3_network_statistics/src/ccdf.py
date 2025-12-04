import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import os    
import powerlaw
from numba import njit, prange
from common.utilities import format_sci

def top_share(panel_df, output_path, start, end, country):

    for year in range(start, end+1):
    
        df_year = panel_df[panel_df['year'] == year].copy()
        
        # Extract one row per firm with their turnover and network_sales
        firms = df_year[['vat', 'turnover', 'network_sales']]

        # Drop firms with missing values
        firms = firms.dropna(subset=['turnover', 'network_sales'])

        # Sort by turnover
        firms = firms.sort_values('turnover', ascending=False).reset_index(drop=True)

        # Rank percentiles
        n = len(firms)
        firms['percentile'] = (firms.index + 1) / n

        # Percentile cutoffs
        cutoffs = [0.10, 0.01, 0.001, 0.0001, 0.00001]

        # Total B2B sales
        total_sales = firms['network_sales'].sum()

        rows = {}
        for c in cutoffs:
            g = firms[firms['percentile'] <= c]
            share = g['network_sales'].sum() / total_sales
            col_name = f"Top {c*100}%"
            rows[col_name] = share

        # Build a one–row table
        table = pd.DataFrame([rows])
        table.to_latex(os.path.join(output_path, f'{year}', f'concentration_net_sales_{country}.tex'), index=False)
        table.to_csv(os.path.join(output_path, f'{year}',f'concentration_net_sales_{country}.csv'), index=False)
        
def concentration_b2b_links(full_df, output_path, start, end, country,
                            percent_list=[0.01, 0.05, 0.10]):
    """
    For each year, computes the share of total B2B value accounted for by
    the top p% of links (ranked by sales_ij), and saves ONE table per year.

    - full_df must contain columns: ['year', 'sales_ij']
    - output_path/year/concentration_b2b_links.(tex|csv)
    """

    for year in range(start, end + 1):
        df_year = full_df[full_df['year'] == year].copy()
        df_year = df_year.dropna(subset=['sales_ij'])
        df_year = df_year[df_year['sales_ij'] > 0]

        if df_year.empty:
            continue  # skip years with no data

        # total B2B sales in that year
        total_m = df_year['sales_ij'].sum()

        # rank links by B2B value descending
        df_year = df_year.sort_values('sales_ij', ascending=False)
        n = len(df_year)

        # build a single row for this year
        result_row = {}
        for p in percent_list:
            k = int(np.ceil(p * n))  # number of links in top p%
            top_value = df_year.iloc[:k]['sales_ij'].sum()
            result_row[f"Top {int(p*100)}%"] = top_value / total_m

        df_out = pd.DataFrame([result_row])

        # save per-year table
        df_out.to_latex(os.path.join(output_path, f'{year}', f"concentration_b2b_links_{country}.tex"),
                        index=False)
        df_out.to_csv(os.path.join(output_path, f'{year}', f"concentration_b2b_links_{country}.csv"),
                      index=False)

    

@njit(parallel=True)
def _search_best_xmin_numba(data_sorted, xmin_candidates, min_tail_size):
    n = data_sorted.size
    m = xmin_candidates.size
    
    # Arrays to store parallel results
    out_alpha = np.zeros(m)
    out_ks = np.zeros(m)
    out_tail_idx = np.zeros(m, dtype=np.int64)

    # Parallel loop over all xmin candidates
    for j in prange(m):
        xmin = xmin_candidates[j]

        # Find start index for the tail
        start_idx = np.searchsorted(data_sorted, xmin)
        n_tail = n - start_idx

        # Not enough tail samples → invalid candidate
        if n_tail < min_tail_size:
            out_alpha[j] = -1.0
            out_ks[j] = 1e308
            out_tail_idx[j] = -1
            continue

        tail = data_sorted[start_idx:n]

        # Compute MLE exponent
        sum_logs = 0.0
        for x in tail:
            sum_logs += np.log(x / xmin)

        if sum_logs <= 0.0:
            out_alpha[j] = -1.0
            out_ks[j] = 1e308
            out_tail_idx[j] = -1
            continue

        alpha = 1.0 + n_tail / sum_logs

        # KS statistic
        ks = 0.0
        for i in range(n_tail):
            F_emp = (i + 1) / n_tail
            F_th = 1.0 - (tail[i] / xmin) ** (1.0 - alpha)
            diff = abs(F_emp - F_th)
            if diff > ks:
                ks = diff

        out_alpha[j] = alpha
        out_ks[j] = ks
        out_tail_idx[j] = start_idx

    # After parallel loop: select best candidate
    best_idx = np.argmin(out_ks)
    return (
        out_alpha[best_idx],
        xmin_candidates[best_idx],
        out_ks[best_idx],
        out_tail_idx[best_idx]
    )

def estimate_powerlaw_tail_numba(
    data,
    xmin_candidates=None,
    min_tail_size=50,
    exhaustive=False,       # True = use all unique values (Stata mode)
    verbose=False,
):
    data = np.asarray(data)
    data = data[np.isfinite(data) & (data > 0)]
    if data.size == 0:
        return None

    data_sorted = np.sort(data)

    # Choose xmin candidates
    if xmin_candidates is None:
        if exhaustive:
            # Stata paretofit mode: every unique value
            xmin_candidates = np.unique(data_sorted)
        else:
            # Fast mode: quantile grid near the tail
            xmin_candidates = np.quantile(data_sorted, np.linspace(0.80, 0.995, 300))
    else:
        xmin_candidates = np.asarray(xmin_candidates)

    xmin_candidates = xmin_candidates[xmin_candidates > 0]

    # Numba parallel search
    alpha, xmin, ks, tail_start_idx = _search_best_xmin_numba(
        data_sorted,
        xmin_candidates,
        min_tail_size
    )

    if tail_start_idx < 0:
        return None

    x_tail = data_sorted[tail_start_idx:]
    tau_ccdf = alpha - 1.0

    if verbose:
        print(f"alpha (PDF): {alpha:.4f}")
        print(f"tau (CCDF): {tau_ccdf:.4f}")
        print(f"xmin: {xmin:.6g}")
        print(f"KS: {ks:.4f}")
        print(f"n_tail: {x_tail.size}")

    return {
        "alpha_pdf": alpha,
        "tau_ccdf": tau_ccdf,
        "xmin": xmin,
        "ks": ks,
        "n_tail": x_tail.size,
        "x_tail": x_tail,
    }

def plot_powerlaw(
    data,
    fit_result=None,
    kind="ccdf",          # "ccdf" or "pdf"
    ax=None,
    tail_only=False,      # if True, show only x >= xmin empirically
    bins=50,              # for PDF
    empirical_kwargs=None,
    fit_kwargs=None,
):
    """
    Plot empirical distribution (CCDF or PDF) and optional fitted power-law line.

    Parameters
    ----------
    data : array-like
        1D array of observations (e.g. influence, centrality, size).
    fit_result : dict or None
        Output of estimate_powerlaw_tail_numba, expected keys:
            - 'alpha_pdf'
            - 'tau_ccdf'
            - 'xmin'
        If None, only empirical distribution is plotted.
    kind : {"ccdf", "pdf"}
        Type of plot:
        - "ccdf": empirical CCDF and Pareto CCDF fit
        - "pdf" : empirical (log-binned) PDF and Pareto PDF fit
    ax : matplotlib Axes or None
        Axis to plot on. If None, uses current axis.
    tail_only : bool
        If True, only plot empirical data for x >= xmin (if fit_result is given).
    bins : int
        Number of bins for the empirical PDF (ignored for CCDF).
    empirical_kwargs : dict or None
        Extra kwargs passed to ax.plot / ax.scatter for empirical series.
    fit_kwargs : dict or None
        Extra kwargs passed to ax.plot for fitted power-law line.

    Returns
    -------
    ax : matplotlib Axes
    """
    data = np.asarray(data)
    data = data[np.isfinite(data) & (data > 0)]
    if data.size == 0:
        raise ValueError("No positive finite data points provided.")

    if ax is None:
        ax = plt.gca()

    if empirical_kwargs is None:
        empirical_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}

    # sensible defaults for empirical dots
    empirical_kwargs.setdefault("marker", ".")
    empirical_kwargs.setdefault("linewidth", 0)
    empirical_kwargs.setdefault("alpha", 0.5)
    empirical_kwargs.setdefault("color", "navy")
    empirical_label = empirical_kwargs.pop("label", "Empirical")

    # sensible defaults for fitted line
    fit_kwargs.setdefault("color", "crimson")
    fit_kwargs.setdefault("linewidth", 2)
    fit_label = fit_kwargs.pop("label", "Power-law fit")

    # If we have a fit result, grab parameters
    xmin = None
    alpha = None
    #tau = None
    p_tail = None

    if fit_result is not None:
        alpha = float(fit_result["alpha_pdf"])
        #tau = float(fit_result["tau_ccdf"])
        xmin = float(fit_result["xmin"])
        # fraction of data in the tail
        p_tail = np.mean(data >= xmin)

    # Possibly restrict to tail-only data for the empirical plot
    if tail_only and (xmin is not None):
        data_emp = data[data >= xmin]
    else:
        data_emp = data

    # ------------------------------------------------------------------
    # CCDF
    # ------------------------------------------------------------------
    if kind.lower() == "ccdf":
        x_sorted = np.sort(data_emp)
        n = x_sorted.size

        # Collapse to unique x's
        unique_x, counts = np.unique(x_sorted, return_counts=True)
        # survivors at each unique x: number of points >= that x
        surv = np.cumsum(counts[::-1])[::-1]
        ccdf = surv / n

        ax.plot(unique_x, ccdf, label=empirical_label, **empirical_kwargs)

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.set_yscale("log")
        ax.set_xlabel("x")
        ax.set_ylabel("CCDF: P(X ≥ x)")

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------
    elif kind.lower() == "pdf":
        # Empirical PDF via histogram
        # Use log-spaced bins for smoother tail behavior
        x_min = data_emp.min()
        x_max = data_emp.max()
        bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), bins + 1)

        counts, edges = np.histogram(data_emp, bins=bin_edges, density=True)
        # midpoints for plotting
        mids = np.sqrt(edges[:-1] * edges[1:])

        # empirical PDF
        ax.plot(mids, counts, label=empirical_label, **empirical_kwargs)

        # fitted PDF: f(x) = p_tail * (alpha-1) * xmin^(alpha-1) * x^{-alpha}
        if fit_result is not None and xmin is not None and alpha is not None:
            x_fit = np.logspace(np.log10(xmin), np.log10(data.max()), 200)
            if p_tail is None:
                p_tail = 1.0
            # For a Pareto(xmin, alpha), conditional on being in the tail,
            # PDF is f_cond(x) = (alpha-1)*xmin^(alpha-1)*x^(-alpha), x >= xmin
            # Unconditional PDF for full data:
            f_fit = p_tail * (alpha - 1.0) * (xmin ** (alpha - 1.0)) * (x_fit ** (-alpha))

            ax.plot(x_fit, f_fit, label=fit_label, **fit_kwargs)

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.set_yscale("log")
        ax.set_xlabel("x")
        ax.set_ylabel("PDF: f(x)")

    else:
        raise ValueError("kind must be 'ccdf' or 'pdf'")

    ax.grid(alpha=0.3)

    return ax

def run_ccdf(panel_df, output_path, start, end, country):
    
    vars = ['turnover', 'network_sales', 'domar', 'outdeg', 'indeg', 'centrality'] 
    
    label_map = {
        'turnover': 'Total sales',
        'network_sales': 'Network sales',
        'outdeg': 'Number of customers',
        'indeg': 'Number of suppliers',
        'domar': 'Domar weight',
        'centrality': 'Bonacich centrality'
    }
    
    for var_name in vars:
        
        if var_name == 'outdeg' or var_name == 'indeg':
            is_discrete = True
        else:
            is_discrete = False
        
        xlabel = label_map.get(var_name, var_name)
                    
        for year in range(start, end+1):

            df_year = panel_df[panel_df['year'] == year].copy()
            var_np = np.array(df_year[var_name].dropna())
                
            # CCDF of the degree distribution
            if is_discrete:
                model = powerlaw.Fit(var_np, discrete=True)
                fit_inf = {
                    "alpha_pdf": model.alpha,
                    "tau_ccdf": model.alpha-1,
                    "xmin": model.xmin,
                    "ks": model.D,
                    "n_tail": model.n_tail,
                    "x_tail": model.data,
                }
            else:
                fit_inf = estimate_powerlaw_tail_numba(
                    var_np,
                    exhaustive=True,  
                    verbose=False,
                )
            plt.figure(figsize=(8,4))
            plot_powerlaw(var_np, fit_inf, kind='ccdf', fit_kwargs={'marker': '.', 'color': 'navy', 'alpha': 0.5})
                        
                        # Add text box with fit information
            textstr = '\n'.join((
                            f'Tail exponent:  {fit_inf['tau_ccdf']:.2f}',
                            f'$x_{{min}}$ = {format_sci(fit_inf['xmin'])}'
                        ))
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            plt.gca().text(0.05, 0.2, textstr, transform=plt.gca().transAxes, fontsize=8,
                                    verticalalignment='top', bbox=props)
                        
            # Set plot labels and save figure
            plt.xlabel(xlabel)
            plt.ylabel(r'$P(X\geq x_{min})$')
            plt.grid(True)
            plt.savefig(os.path.join(output_path, f'{year}', 'CCDF', f'ccdf_{year}_{var_name}_{country}.png'), dpi=300, bbox_inches='tight')
            plt.close()

# Master function
def master_CCDF(full_df, panel_df, output_path, start, end, country):
    
    run_ccdf(panel_df, output_path, start, end, country)
    
    top_share(panel_df, output_path, start, end, country)
    
    concentration_b2b_links(full_df, output_path, start, end, country)

            

            
    

        
    
    