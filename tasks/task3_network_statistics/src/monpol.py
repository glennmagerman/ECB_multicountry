import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_irf_up_classes(irf_wide,
                        se_wide):

    horizons = irf_wide.index.to_numpy()
    up_classes = list(irf_wide.columns)

    n_classes = len(up_classes)

    fig, axes = plt.subplots(2, 3, figsize=(18, 6))
    axes = axes.flatten()

    global_min, global_max = float("inf"), float("-inf")

    for idx, up in enumerate(up_classes):
        ax = axes[idx]

        irfs = irf_wide[up].to_numpy() * 100
        ses  = se_wide[up].to_numpy() * 100

        ci_upper_95 = irfs + 1.96 * ses
        ci_lower_95 = irfs - 1.96 * ses
        ci_upper_68 = irfs + ses
        ci_lower_68 = irfs - ses

        # Update global y-limits
        global_min = min(global_min, ci_lower_95.min())
        global_max = max(global_max, ci_upper_95.max())

        # Plot IRF
        ax.plot(horizons, irfs)

        # 95% CI
        ax.fill_between(horizons, ci_lower_95, ci_upper_95,
                        alpha=0.1, label="95% CI", color="blue")
        # 68% CI
        ax.fill_between(horizons, ci_lower_68, ci_upper_68,
                        alpha=0.2, label="68% CI", color="navy")

        # Zero line
        ax.axhline(0, color="black", linewidth=1, linestyle="--")

        ax.set_title(f"Upstreamness class = {up}")
        ax.set_xlabel("Quarters")
        ax.set_ylabel("Response (%)")

        ax.grid(True, linestyle="--", alpha=0.7)

        step = 1
        ax.set_xticks(np.arange(horizons.min(), horizons.max() + 1, step))

        ax.legend()

    for ax in axes:
        ax.set_ylim(global_min, global_max)

    for idx in range(n_classes, len(axes)):
        fig.delaxes(axes[idx])

    plt.subplots_adjust(hspace=0.4, wspace=0.3)

def add_tot_effect_and_se(
    panel_expanded: pd.DataFrame,
    vcov: pd.DataFrame,
    var,
    shock_var: str = "std_MP_median",
    interaction: str = "upstreamness",
    beta_col: str = "beta_sales",
    delta_col: str = "delta_sales",
    z_col: str = "upstreamness",
    h_col: str = "h",
):

    vcov.index   = vcov.index.astype(str)
    vcov.columns = vcov.columns.astype(str)

    beta_names = [name for name in vcov.index
                  if name.startswith(f"{shock_var}_h")]
    delta_names = [name for name in vcov.index
                   if name.startswith(f"{shock_var}_{interaction}_h")]

    # Sort by horizon number (based on the part after `_h`)
    def sort_by_h(names):
        return sorted(names, key=lambda s: int(s.split("_h")[-1]))

    beta_names  = sort_by_h(beta_names)
    delta_names = sort_by_h(delta_names)

    # Sanity check (optional)
    if len(beta_names) != len(delta_names):
        raise ValueError("Number of beta and delta horizons do not match.")

    H = len(beta_names) - 1  

    var_beta       = np.diag(vcov.loc[beta_names,  beta_names])
    var_delta      = np.diag(vcov.loc[delta_names, delta_names])
    cov_beta_delta = np.diag(vcov.loc[beta_names,  delta_names])

    var_parts = pd.DataFrame({
        h_col:       range(H + 1),
        "var_beta":  var_beta,
        "var_delta": var_delta,
        "cov_beta_delta": cov_beta_delta,
    })

    panel_h = panel_expanded.merge(var_parts, on=h_col, how="left")

    z = panel_h[z_col]

    panel_h[f"tot_eff_{var}"] = panel_h[beta_col] + panel_h[delta_col] * z

    panel_h[f"var_tot_eff_{var}"] = (
        panel_h["var_beta"]
        + (z ** 2) * panel_h["var_delta"]
        + 2 * z * panel_h["cov_beta_delta"]
    )

    panel_h[f"se_tot_eff_{var}"] = np.sqrt(panel_h[f"var_tot_eff_{var}"])

    return panel_h

def plot_irfs(panel_h, output_path, fig_name, var):
    
    panel = panel_h.copy()
    panel["upstreamness"] = panel["upstreamness"].clip(upper=20)
    
    # average upstreamness by class
    avg_up = (
        panel
        .groupby("up_class")["upstreamness"]
        .mean()
        .rename("avg_up_class")
        .reset_index()
    )

    var_parts = (
        panel[["h", f"beta_{var}", f"delta_{var}", "var_beta", "var_delta", "cov_beta_delta"]]
        .drop_duplicates(subset="h")
        .reset_index(drop=True)
    )

    var_parts["key"] = 1
    avg_up["key"] = 1
    class_irf = var_parts.merge(avg_up, on="key").drop(columns="key")

    zbar = class_irf["avg_up_class"]

    class_irf["tot_eff_class"] = class_irf[f"beta_{var}"] + class_irf[f"delta_{var}"] * zbar

    class_irf["var_tot_eff_class"] = (
        class_irf["var_beta"]
        + (zbar ** 2) * class_irf["var_delta"]
        + 2 * zbar * class_irf["cov_beta_delta"]
    )

    class_irf["se_tot_eff_class"] = np.sqrt(class_irf["var_tot_eff_class"])
    irf_wide = class_irf.pivot(
        index="h",
        columns="up_class",
        values="tot_eff_class"
    ).sort_index()

    se_wide = class_irf.pivot(
        index="h",
        columns="up_class",
        values="se_tot_eff_class"
    ).sort_index()
    
    # plot subplots (one per class, with CIs)
    plot_irf_up_classes(
        irf_wide=irf_wide,
        se_wide=se_wide
    )

    out_dir = os.path.join(output_path, 'all_years','monpol')
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, fig_name)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def avg_contribution_byclass(panel_df, panel_h, var, output_path, country):
    
    panel_df = panel_df.sort_values(["vat","year"])
    panel_df["total_turnover_year"] = panel_df.groupby("year")["turnover"].transform("sum")
    panel_df["sales_share"] = panel_df["turnover"] / panel_df["total_turnover_year"]
    panel_h = panel_h.merge(
        panel_df[["vat","year","sales_share"]],
        on=["vat","year"],
        how="left"
    )
    
    # total effects and contribution by horizon-year
    agg_year = (
        panel_h
        .groupby(["year", "h"])
        .apply(lambda g: (g["sales_share"] * g[f"tot_eff_{var}"]).sum())
        .rename("agg_eff")
        .reset_index()
    )
    
    contrib_year = (
        panel_h
        .groupby(["year", "h", "up_class"])
        .apply(lambda g: (g["sales_share"] * g[f"tot_eff_{var}"]).sum())
        .rename("contrib")
        .reset_index()
    )
    
    contrib_year = contrib_year.merge(
        agg_year,
        on=["year", "h"],
        how="left"
    )

    contrib_year["share"] = contrib_year["contrib"] / contrib_year["agg_eff"]
    
    # average effects and contribution by horizon across years
    share_overall = (
        contrib_year
        .groupby(["h", "up_class"])["share"]
        .mean()
        .rename("share_overall")
        .reset_index()
    )

    share_wide = share_overall.pivot(index="h", columns="up_class", values="share_overall")
    col_means = share_wide.mean(axis=0)
    interval_labels = {
        1: r"$U = 1$",
        2: r"$1 < U \leq 2$",
        3: r"$2 < U \leq 3$",
        4: r"$3 < U \leq 4$",
        5: r"$U > 4$"
    }

    classes = sorted(col_means.index)
    means_row = [col_means[c] for c in classes]
    col_headers = [interval_labels[c] for c in classes]

    formatted = [f"{m:.2f}" for m in means_row] 

    latex = r"\begin{table}[H]" "\n" \
            r"\centering" "\n" \
            r"\begin{tabular}{l" + "c"*len(classes) + "}" "\n" \
            r"\toprule" "\n" \
            r" & " + " & ".join(col_headers) + r" \\" "\n" \
            r"\midrule" "\n" \
            r"Avg.\ contribution across horizons & " + " & ".join(formatted) + r" \\" "\n" \
            r"\bottomrule" "\n" \
            r"\end{tabular}" "\n" \
            r"\caption{Contribution of upstreamness classes to the total response of sales to a monetary policy shock.}" "\n" \
            r"\label{tab:upstream_contrib}" "\n" \
            r"\end{table}"

    tex_path = os.path.join(
                output_path, 'all_years','monpol',
                f"avg_contribution_byup_{var}_{country}.tex"
            )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)
        
def avg_contribution_by_percentiles(panel_df, panel_h, var, output_path, country):

    # Sales shares by year
    panel_df = panel_df.sort_values(["vat", "year"])
    panel_df["total_turnover_year"] = panel_df.groupby("year")["turnover"].transform("sum")
    panel_df["sales_share"] = panel_df["turnover"] / panel_df["total_turnover_year"]

    panel_h = panel_h.merge(
        panel_df[["vat", "year", "sales_share"]],
        on=["vat", "year"],
        how="left"
    )

    # Total (sales-weighted) effect by year and horizon
    agg_year = (
        panel_h
        .groupby(["year", "h"])
        .apply(lambda g: (g["sales_share"] * g[f"tot_eff_{var}"]).sum())
        .rename("agg_eff")
        .reset_index()
    )

    # Define percentile thresholds (fractions of the upper tail)
    # top 10%, top 1%, top 0.01%, top 0.001%
    tail_fracs = [0.10, 0.01, 0.0001, 0.00001]

    # Compute global cutoffs for upstreamness (over all firm-years)
    cutoffs = {
        frac: panel_df["upstreamness"].quantile(1 - frac)
        for frac in tail_fracs
    }

    # For each tail, compute average share of aggregate effect
    avg_shares = {}

    for frac in tail_fracs:
        cutoff = cutoffs[frac]

        # restrict to firms in the top frac of upstreamness
        mask = panel_h["upstreamness"] >= cutoff
        panel_tail = panel_h[mask].copy()

        if panel_tail.empty:
            avg_shares[frac] = np.nan
            continue

        # sales-weighted contribution of this tail by (year, h)
        contrib_tail_year = (
            panel_tail
            .groupby(["year", "h"])
            .apply(lambda g: (g["sales_share"] * g[f"tot_eff_{var}"]).sum())
            .rename("contrib")
            .reset_index()
        )

        # merge with total aggregate effect to get shares
        contrib_tail_year = contrib_tail_year.merge(
            agg_year,
            on=["year", "h"],
            how="left"
        )

        contrib_tail_year = contrib_tail_year.dropna(subset=["agg_eff"])
        contrib_tail_year = contrib_tail_year[contrib_tail_year["agg_eff"] != 0]

        if contrib_tail_year.empty:
            avg_shares[frac] = np.nan
            continue

        contrib_tail_year["share"] = contrib_tail_year["contrib"] / contrib_tail_year["agg_eff"]

        # First average across years for each horizon
        share_over_h = (
            contrib_tail_year
            .groupby("h")["share"]
            .mean()
            .reset_index()
        )

        # Then average across horizons
        avg_share = share_over_h["share"].mean()
        avg_shares[frac] = avg_share

    # Build LaTeX table
    # Labels for columns
    perc_labels = {
        0.10: r"Top 10\%",
        0.01: r"Top 1\%",
        0.0001: r"Top 0.01\%",
        0.00001: r"Top 0.001\%",
    }

    fracs_ordered = [0.10, 0.01, 0.0001, 0.00001]
    col_headers = [perc_labels[f] for f in fracs_ordered]

    # Format as percentages (shares × 100)
    formatted = []
    for f in fracs_ordered:
        v = avg_shares.get(f, np.nan)
        if np.isnan(v):
            formatted.append("--")
        else:
            formatted.append(f"{100 * v:.1f}\\%")

    latex = (
        r"\begin{table}[H]" "\n"
        r"\centering" "\n"
        r"\begin{tabular}{l" + "c" * len(fracs_ordered) + "}" "\n"
        r"\toprule" "\n"
        r" & " + " & ".join(col_headers) + r" \\" "\n"
        r"\midrule" "\n"
        r"Avg.\ contribution across horizons & " + " & ".join(formatted) + r" \\" "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        rf"\caption{{Contribution of extreme upstreamness percentiles to the total response of {var} "
        r"to a monetary policy shock.}}" "\n"
        rf"\label{{tab:upstream_tail_contrib_{var}_{country}}}" "\n"
        r"\end{table}"
    )

    tex_path = os.path.join(
        output_path, 'all_years','monpol',
        f"avg_contribution_bypercentiles_{var}_{country}.tex"
    )
    os.makedirs(os.path.dirname(tex_path), exist_ok=True)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)
        
def plot_percentile_contrib_by_horizon_delta(
    panel_df: pd.DataFrame,
    panel_h: pd.DataFrame,
    var: str,
    output_path: str,
    country: str,
    percentiles=(0.10, 0.01, 0.0001, 0.00001),
):

    panel_df = panel_df.sort_values(["vat", "year"]).copy()
    panel_df["total_turnover_year"] = panel_df.groupby("year")["turnover"].transform("sum")
    panel_df["sales_share"] = panel_df["turnover"] / panel_df["total_turnover_year"]

    # Weights and upstreamness at firm-year level
    w = panel_df["sales_share"].to_numpy()
    z = panel_df["upstreamness"].to_numpy()

    # Global sums (over all firm-years).
    S0 = np.nansum(w)
    S1 = np.nansum(w * z)

    # Define tails
    cutoffs = {p: panel_df["upstreamness"].quantile(1 - p) for p in percentiles}

    S0_p = {}
    S1_p = {}
    for p, c in cutoffs.items():
        mask = (z >= c)
        w_p = w[mask]
        z_p = z[mask]
        S0_p[p] = np.nansum(w_p)
        S1_p[p] = np.nansum(w_p * z_p)

    # Extract beta, delta, and their vcov by horizon 
    var_parts = (
        panel_h[["h", f"beta_{var}", f"delta_{var}", "var_beta", "var_delta", "cov_beta_delta"]]
        .drop_duplicates(subset="h")
        .sort_values("h")
        .reset_index(drop=True)
    )

    horizons = var_parts["h"].to_numpy()
    beta_h   = var_parts[f"beta_{var}"].to_numpy()
    delta_h  = var_parts[f"delta_{var}"].to_numpy()
    var_b    = var_parts["var_beta"].to_numpy()
    var_d    = var_parts["var_delta"].to_numpy()
    cov_bd   = var_parts["cov_beta_delta"].to_numpy()

    H = len(horizons)
    T = len(percentiles)

    share_mean = np.empty((H, T), dtype=float)
    share_se   = np.empty((H, T), dtype=float)

    # Delta-method variance 
    tails = list(percentiles)

    for j, p in enumerate(tails):
        s0_p = S0_p[p]
        s1_p = S1_p[p]

        for k in range(H):
            b = beta_h[k]
            d = delta_h[k]

            D = b * S0 + d * S1
            N_p_h = b * s0_p + d * s1_p

            if D == 0:
                # If the aggregate effect is exactly zero, the share is undefined.
                # Here we set to NaN and skip SE.
                share_mean[k, j] = np.nan
                share_se[k, j] = np.nan
                continue

            S_h = N_p_h / D
            share_mean[k, j] = S_h

            dS_dbeta = (s0_p * D - N_p_h * S0) / (D**2)
            dS_ddelta = (s1_p * D - N_p_h * S1) / (D**2)

            # Delta-method variance
            vb = var_b[k]
            vd = var_d[k]
            cbd = cov_bd[k]

            var_S = (
                dS_dbeta**2 * vb
                + dS_ddelta**2 * vd
                + 2.0 * dS_dbeta * dS_ddelta * cbd
            )

            var_S = max(var_S, 0.0)
            share_se[k, j] = np.sqrt(var_S)

    # 95% CIs
    share_lo = share_mean - 1.96 * share_se
    share_hi = share_mean + 1.96 * share_se

    # Plot
    mean_pct = share_mean * 100.0
    lo_pct   = share_lo * 100.0
    hi_pct   = share_hi * 100.0

    n_panels = T
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12,6))
    axes = np.array(axes).reshape(-1)

    tail_labels = {
        0.10: "Top 10%",
        0.01: "Top 1%",
        0.0001: "Top 0.01%",
        0.00001: "Top 0.001%",
    }

    for j, p in enumerate(tails):
        ax = axes[j]

        m = mean_pct[:, j]
        lo = lo_pct[:, j]
        hi = hi_pct[:, j]

        mask = ~np.isnan(m)
        if not np.any(mask):
            ax.set_title(tail_labels.get(p, f"Top {p*100:.3f}%"))
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        x = horizons[mask]
        m = m[mask]
        lo = lo[mask]
        hi = hi[mask]

        yerr = np.vstack([m - lo, hi - m])

        ax.errorbar(
            x, m, yerr=yerr,
            fmt="o",                      
            capsize=3,
            ecolor='black',              
            markerfacecolor="navy",      
            markeredgecolor="black",
            linestyle="none"
        )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(tail_labels.get(p, f"Top {p*100:.3f}%"))
        ax.set_xlabel("Quarter")
        ax.set_xticks(horizons) 
        ax.set_ylabel("Contribution to total effect (%)")
        ax.grid(True, linestyle="--", alpha=0.6)

    # Remove unused subplots
    for k in range(n_panels, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()

    out_dir = os.path.join(output_path, 'all_years',"monpol")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"contrib_percentiles_{var}_{country}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

def master_monpol(panel_df, input_path, output_path, country):
    
    irfs_sales = pd.read_csv(os.path.join(input_path, 'irfs_int_up_quarterly.csv')).rename(columns={'Unnamed: 0': 'h'})
    irfs_prices = pd.read_csv(os.path.join(input_path, 'irfs_p_int_up_quarterly.csv')).rename(columns={'Unnamed: 0': 'h'})
    vcov_sales = pd.read_csv(os.path.join(input_path, 'vcov_bs_int_up_quarterly.csv'), index_col=0)
    vcov_prices = pd.read_csv(os.path.join(input_path, 'vcov_p_bs_int_up_quarterly.csv'), index_col=0)
    
    h = 12
    horizons = np.arange(h+1)

    # sort once
    panel_df = panel_df.sort_values(['vat', 'year']).reset_index(drop=True)
    panel_df["upstreamness"] = (
        panel_df.groupby("vat")["upstreamness"].shift(1)
    )

    for var in ["sales", "prices"]:
        
        # pick the right IRFs / vcov and figure name
        if var == "sales":
            irfs = irfs_sales
            vcov = vcov_sales
            fig_name = f"irf_sales_up_class_{country}.png"
        else:  # "prices"
            irfs = irfs_prices
            vcov = vcov_prices
            fig_name = f"irf_prices_up_class_{country}.png"

        # expand panel to firm-year-h
        panel_expanded = panel_df.loc[panel_df.index.repeat(len(horizons))].copy()
        panel_expanded['h'] = np.tile(horizons, len(panel_df))  # add horizon col

        # merge beta and delta for this outcome
        panel_expanded = panel_expanded.merge(
            irfs[['h','std_MP_median', 'std_MP_median_upstreamness']].rename(
                columns={
                    'std_MP_median':f'beta_{var}',
                    'std_MP_median_upstreamness': f'delta_{var}'
                }),
            on='h', how='left'
        )
        
        # delta-method firm-level total effects
        panel_h = add_tot_effect_and_se(panel_expanded, vcov, var, beta_col=f'beta_{var}', delta_col=f'delta_{var}')
        
        # upstreamness classes
        panel_h = panel_h.dropna(subset=['upstreamness']).copy()
        panel_h["up_class"] = np.ceil(panel_h["upstreamness"]).astype(int)
        panel_h.loc[panel_h["up_class"] > 5, "up_class"] = 5
        
        plot_irfs(panel_h, output_path, fig_name, var=var)
        
        avg_contribution_byclass(panel_df, panel_h, var, output_path, country)
        avg_contribution_by_percentiles(panel_df, panel_h, var, output_path, country)
        






        
        

        