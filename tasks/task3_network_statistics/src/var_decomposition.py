import os
import pyfixest as pf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.utilities import demean_variable_in_df

def extract_two_way_fe(df, yvar, fe_vars, res,
                             resid_col="resid",
                             max_iter=1000,
                             tol=1e-10):
    """

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe used in the regression (same rows, same ordering).
        Must contain a column resid_col with the regression residuals.
    yvar : str
        Name of the dependent variable column in df.
    fe_vars : list[str]
        List of fixed-effect variables, e.g. ["vat_i", "vat_j"].
    res : pyfixest result object (feols)
        The fitted pyfixest model. Must have:
            - res.coef  (pandas Series of coefficients)
        (res.resid is NOT used; residuals are taken from df[resid_col].)
    resid_col : str
        Name of the column in df containing the residuals.
    max_iter : int
        Maximum number of full sweeps over all FE dimensions.
    tol : float
        Convergence tolerance on the change in residual_fe standard deviation.

    Returns
    -------
    fe_contrib_obs : dict[str, np.ndarray]
        For each FE variable h in fe_vars, an array of length N with the
        per-observation contribution of that FE (like Stata's a(h=...)).
    fe_levels : dict[str, pd.Series]
        For each FE variable h, a Series mapping each level to its FE coefficient
        (alpha_g), i.e. constant within group.
    residual_fe : np.ndarray
        Final leftover after the decomposition (should be ~0 if converged).
    total_fe : np.ndarray
        The total FE part y - Xβ - u that is being decomposed.
    """

    N = len(df)
    y = df[yvar].to_numpy(dtype=float)

    # 1. Compute Xβ from pyfixest coefficients
    beta = res._beta_hat
    if beta.size > 0:
        # We assume the covariate names in beta.index exist as columns in df
        X = df[beta.index].to_numpy(dtype=float)
        xb = X @ beta.to_numpy(dtype=float)
    else:
        xb = np.zeros(N, dtype=float)

    # 2. Residuals: take them directly from the dataframe
    uhat = df[resid_col].to_numpy(dtype=float)

    # 3. Total FE part: FE_total = y - Xβ - u
    total_fe = y - xb - uhat

    # 4. Precompute integer codes for each FE variable for fast group means
    codes_dict = {}
    cats_dict = {}
    ng_dict = {}
    for fe in fe_vars:
        cat = df[fe].astype("category")
        codes = cat.cat.codes.to_numpy()  # 0..G_h-1
        codes_dict[fe] = codes
        cats_dict[fe] = cat.cat.categories
        ng_dict[fe] = codes.max() + 1

    # 5. Initialize contributions and residual_fe
    fe_contrib_obs = {fe: np.zeros(N, dtype=float) for fe in fe_vars}
    residual_fe = total_fe.copy()
    prev_std = np.std(residual_fe)

    # 6. Alternating projections (iterative demeaning)
    for it in range(max_iter):
        for fe in fe_vars:
            codes = codes_dict[fe]
            ng = ng_dict[fe]

            # group sums and counts of current residual_fe
            sums = np.bincount(codes, weights=residual_fe, minlength=ng)
            counts = np.bincount(codes, minlength=ng)
            means = sums / counts

            # contribution from this FE dimension for each observation
            contrib = means[codes]

            # accumulate into that FE's contribution and update residual_fe
            fe_contrib_obs[fe] += contrib
            residual_fe -= contrib

        cur_std = np.std(residual_fe)
        if prev_std - cur_std < tol:
            break
        prev_std = cur_std

    # 7. Compute per-level FE coefficients (group means of contributions)
    fe_levels = {}
    for fe in fe_vars:
        codes = codes_dict[fe]
        arr = fe_contrib_obs[fe]
        ng = ng_dict[fe]

        sums = np.bincount(codes, weights=arr, minlength=ng)
        counts = np.bincount(codes, minlength=ng)
        means = sums / counts

        cats = cats_dict[fe]
        fe_levels[fe] = pd.Series(means[: len(cats)], index=cats, name=f"{fe}_fe")

    return fe_contrib_obs, fe_levels, residual_fe, total_fe
    
def check_identity(df, lhs, rhs_vars, tol=1e-5, raise_error=False, verbose=True):

    # compute RHS
    rhs = df[rhs_vars].sum(axis=1)

    # compute difference
    diff = df[lhs] - rhs

    # test identity
    if np.allclose(diff, 0, atol=tol):
        if verbose:
            print(f"Identity holds: {lhs} = sum({rhs_vars})")
        return True
    else:
        max_err = np.max(np.abs(diff))
        if verbose:
            print(f"Identity FAILED for {lhs} = sum({rhs_vars})")
            print(f"   Max |LHS − RHS| = {max_err:.3e}")

        if raise_error:
            raise ValueError(f"Identity failed in variance decomposition: {lhs} != sum({rhs_vars})")

        return False
    
def report_variance_fe(fe1, fe2, mod):
    
    var_fe1 = np.var(fe1)
    var_fe2 = np.var(fe2)
    var_fe1_fe2 = np.var(fe1+fe2)
    cov_fe1_fe2 = np.cov(fe1, fe2)[0,1]
    df = mod._data.copy()
    y = df[mod._depvar].to_numpy()
    resid = mod.resid()
    
    SSR = (resid**2).sum()
    SST = ((y - y.mean())**2).sum()
    R2 = 1-SSR/SST
    N = mod._N
    k=mod._k_fe.sum(0)
    adj_R2 = 1-((1-R2)*(N-1)/(N-k-1))
    
    row = (
        r"$\ln m_{ij}$ & "
        f"{N:,.0f} & "
        f"{var_fe1 / var_fe1_fe2:.2f} & "
        f"{var_fe2 / var_fe1_fe2:.2f} & "
        f"{2 * cov_fe1_fe2 / var_fe1_fe2:.2f} & "
        f"{R2:.2f} & "
        f"{adj_R2:.2f} \\\\"
    )

    latex = rf"""
        \begin{{table}}[htbp]
        \centering
        \caption{{{'Buyer and Seller Effects'}}}
        \label{{{'tab:var_fixed_eff'}}}
        \begin{{tabular}}{{l r c c c c c}}
        \toprule
        & $N$ & 
        $\frac{{\mathrm{{var}}(\ln \psi_i)}}{{\mathrm{{var}}(\ln \psi_i + \ln \theta_j)}}$ &
        $\frac{{\mathrm{{var}}(\ln \theta_j)}}{{\mathrm{{var}}(\ln \psi_i + \ln \theta_j)}}$ &
        $\frac{{2\mathrm{{cov}}(\ln \psi_i,\ln \theta_j)}}{{\mathrm{{var}}(\ln \psi_i + \ln \theta_j)}}$ &
        $R^2$ & Adjusted $R^2$ \\\\
        \midrule
        {row}
        \bottomrule
        \end{{tabular}}
        \end{{table}}
        """
    
    return latex

def tw_fe(full_df, year, output_path, country):
    
    # focus on year 2019 for now
    df_year = full_df[full_df['year'] == year].copy()
    
    df_year['ln_sales_ij'] = np.log(df_year['sales_ij'].where(df_year['sales_ij'] > 0))
    
    # TWFE regression
    model = pf.feols('ln_sales_ij ~ 1 | vat_i + vat_j', data=df_year, fixef_rm='singleton')
    
    # extract model objects
    new_df = model._data.copy() # extract data without singletons
    new_df['ln_omega_ij'] = model.resid()
    fe_contrib_obs, fe_levels, residual_fe, total_fe = extract_two_way_fe(
        df=new_df,
        yvar="ln_sales_ij",
        fe_vars=["vat_i", "vat_j"],
        res=model,              
        resid_col="ln_omega_ij",
    )
    
    # test: LHS - RHS = 0
    new_df['total_fe'] = total_fe
    rhs_vars = ["ln_omega_ij", "total_fe"]
    check_identity(new_df, 'ln_sales_ij', rhs_vars)
    
    # produce table with (co)variances of FEs
    ln_psi_i = fe_contrib_obs['vat_i']
    ln_theta_j = fe_contrib_obs['vat_j']
    latex = report_variance_fe(ln_psi_i, ln_theta_j, model)
    with open(os.path.join(output_path, f'{year}', 'var_decomp',f'var_fe_{country}.tex'), "w", encoding="utf-8") as f:
        f.write(latex)
    
    # transform logs to levels
    new_df["psi_i"] = np.exp(ln_psi_i)
    new_df["theta_j"] = np.exp(ln_theta_j)
    new_df['omega_ij'] = np.exp(new_df['ln_omega_ij'])

    return new_df


def create_components(new_df, panel_df, year):

    # Recompute firm-level outdegree and network sales from new data
    outdeg = (
        new_df.groupby("vat_i")["vat_j"]
              .nunique()
              .rename("outdeg")
    )

    network_sales = (
        new_df.groupby("vat_i")["sales_ij"]
              .sum()
              .rename("network_sales")
    )

    new_df = (
        new_df
        .merge(outdeg, on="vat_i", how="left")
        .merge(network_sales, on="vat_i", how="left")
    )

    # Average customer capability (geometric mean of theta_j)
    new_df["geom_obj"] = new_df["theta_j"] ** (1 / new_df["outdeg"])
    avg_theta_i = (
        new_df.groupby("vat_i")["geom_obj"]
              .prod()
              .reset_index()
              .rename(columns={"geom_obj": "avg_theta_i"})
    )
    new_df = new_df.merge(avg_theta_i, on="vat_i", how="left")

    #  Omega_i 
    new_df["Omega_i"] = (
        new_df["omega_ij"] * new_df["theta_j"]
        / (new_df["avg_theta_i"] * new_df["outdeg"])
    )
    Omega_i = (
        new_df.groupby("vat_i")["Omega_i"]
              .sum()
              .reset_index()
    )

    # Assemble firm-level regression panel from new_df
    reg_panel = (
        Omega_i
        .merge(
            new_df[["vat_i", "psi_i", "avg_theta_i", "outdeg", "network_sales"]]
            .drop_duplicates(subset="vat_i"),
            on="vat_i",
            how="left",
        )
        .rename(columns={"vat_i": "vat"})
    )

    # Add turnover and nace from panel_df (firm totals)
    panel_df_year = panel_df[panel_df["year"] == year].copy()
    reg_panel = reg_panel.merge(
        panel_df_year[["vat", "turnover", "nace", "nace2d"]],
        on="vat",
        how="left",
    )

    # final demand beta_i
    reg_panel["beta_i"] = reg_panel["turnover"] / reg_panel["network_sales"]
    
    vars = ['psi_i', 'outdeg', 'beta_i', 'avg_theta_i', 'Omega_i', 'turnover']
    for var in vars:
        reg_panel[f'ln_{var}'] = np.log(reg_panel[var].where(reg_panel[var] > 0))
        reg_panel = reg_panel.groupby('nace').filter(lambda x: len(x) >= 5) # drop NACE codes with less than 5 observations
        reg_panel[f'ln_{var}_dem'] = demean_variable_in_df(f'ln_{var}', 'nace', reg_panel)
        
    # test: LHS - RHS = 0
    rhs_vars = ["ln_psi_i", "ln_outdeg", "ln_avg_theta_i","ln_Omega_i", "ln_beta_i"]
    check_identity(reg_panel, 'ln_turnover', rhs_vars)

    return reg_panel

def var_decomposition(reg_panel, year, output_path, country, by_sector=False):

    # Variance decomposition
    labels = {
        "beta_i":       r"Relative final demand $\ln \beta_i$",
        "psi_i":        r"Upstream $\ln \psi_i$",
        "outdeg":       r"# Customers $\ln n_i^c$",
        "avg_theta_i":  r"Avg customer capability $\ln \bar{\theta}_i$",
        "Omega_i":      r"Customer interaction $\ln \Omega_i^c$",
    }

    if not by_sector:
        rows = []

        vars = ['psi_i', 'outdeg', 'beta_i', 'avg_theta_i', 'Omega_i', 'turnover']
        for var in vars:
            if var == "turnover":
                continue  # skip ln_turnover itself

            y = f"ln_{var}_dem"
            model = pf.feols(f"{y} ~ ln_turnover_dem", reg_panel, fixef_rm="singleton")

            beta = model.coef()["ln_turnover_dem"]
            se   = model.se()["ln_turnover_dem"]

            rows.append({
                "Component": labels.get(var, var),
                "Share": beta,
                "SE": se,
            })

        table = pd.DataFrame(rows)
        table["Estimate (SE)"] = table.apply(
            lambda r: f"{r['Share']:.2f} ({r['SE']:.2f})",
            axis=1,
        )
        table = table[["Component", "Estimate (SE)"]]
        latex = table.to_latex(
            index=False,
            escape=False,          
            column_format="l c",   
            caption=r"Firm size decomposition ($\ln S_i$).",
            label="tab:firm_size_decomp",
        )
        with open(os.path.join(output_path, f'{year}', 'var_decomp',f'var_decomp_{country}.tex'), "w", encoding="utf-8") as f:
            f.write(latex)
            
        fig, ax = plot_firm_size_decomposition(reg_panel, n_bins=20)
        plt.savefig(os.path.join(output_path, f'{year}', 'var_decomp',f'firm_size_decomp_{country}.png'), dpi=300)
        plt.close()
            
    else:
        
        rows = []

        for sec in sorted(reg_panel["nace2d"].unique()):
            df_sec = reg_panel[reg_panel["nace2d"] == sec].copy()

            # If too small, skip sector entirely
            if len(df_sec) < 5:
                continue

            sector_row = {"NACE2": sec}

            # We will fill N after the first regression (post-singleton N)
            N_recorded = False

            for var in ["psi_i", "outdeg", "avg_theta_i", "Omega_i", "beta_i"]:

                y = f"ln_{var}_dem"

                # Run sector-level regression with singleton removal
                model = pf.feols(f"{y} ~ ln_turnover_dem", df_sec, fixef_rm="singleton")

                # Record N after singleton removal (correct)
                if not N_recorded:
                    sector_row["N"] = int(model._N)
                    N_recorded = True

                beta = model.coef()["ln_turnover_dem"]
                se   = model.se()["ln_turnover_dem"]

                sector_row[labels[var]] = f"{beta:.2f} ({se:.2f})"

            rows.append(sector_row)

        # Build final table
        table = pd.DataFrame(rows)
        ordered_cols = ["NACE2", "N"] + list(labels.values())
        table = table[ordered_cols]

        latex = table.to_latex(
            index=False,
            escape=False,
            column_format="l r " + "c " * (len(labels)),
            caption=r"Sector-level firm size decomposition ($\ln S_i$).",
            label="tab:firm_size_decomp_sector",
        )

        with open(os.path.join(output_path, f"{year}", 'var_decomp',f"var_decomp_sector_{country}.tex"),
                "w", encoding="utf-8") as f:
            f.write(latex)
            
        
    
def plot_firm_size_decomposition(reg_panel, n_bins=20):
    # 1. Create equal-sized bins in ln_turnover
    reg_panel = reg_panel.copy()
    reg_panel["bin"] = pd.qcut(
        reg_panel["ln_turnover_dem"],
        q=n_bins,
        labels=False,
        duplicates="drop"
    )

    # 2. Compute bin means of ln variables
    cols = [
        "ln_turnover_dem",
        "ln_psi_i_dem",
        "ln_outdeg_dem",
        "ln_avg_theta_i_dem",
        "ln_Omega_i_dem",
        "ln_beta_i_dem",
    ]
    gb = reg_panel.groupby("bin")[cols].mean()

    # 3. Convert to levels for plotting (as in the figure)
    S = np.exp(gb["ln_turnover_dem"])
    psi = np.exp(gb["ln_psi_i_dem"])
    n_c = np.exp(gb["ln_outdeg_dem"])
    theta_bar = np.exp(gb["ln_avg_theta_i_dem"])
    Omega = np.exp(gb["ln_Omega_i_dem"])
    beta = np.exp(gb["ln_beta_i_dem"])

    # 4. Plot: log–log binned scatter
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(S, psi, marker="o", linestyle="none", label=r"$\psi_i$")
    ax.plot(S, n_c, marker="o", linestyle="none", label=r"$n_i^c$")
    ax.plot(S, theta_bar, marker="o", fillstyle="none",
            linestyle="none", label=r"$\bar{\theta}_i$")
    ax.plot(S, Omega, marker="D", linestyle="none", label=r"$\Omega_i^c$")
    ax.plot(S, beta, marker="^", linestyle="none", label=r"$\beta_i$")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Turnover (S)")
    ax.legend()

    fig.tight_layout()
    return fig, ax   
    
#####
def master_var_decomp(full_df, panel_df, output_path, start, end, country):
    
    year = 2019
    new_df = tw_fe(full_df, year, output_path, country)
    
    reg_panel = create_components(new_df, panel_df, year)
    
    var_decomposition(reg_panel, year, output_path, country)
    var_decomposition(reg_panel, year, output_path, country, by_sector=True)
