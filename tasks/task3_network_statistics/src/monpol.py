import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.utilities import save_graph_data

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
    interaction: str = "avg_upstreamness",
    beta_col: str = "beta_sales",
    delta_col: str = "delta_sales",
    z_col: str = "avg_upstreamness",
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
    save_graph_data(
        output_path=output_path,
        file_name=f"{os.path.splitext(fig_name)[0]}_data",
        data=class_irf[["h", "up_class", "tot_eff_class", "se_tot_eff_class"]],
        year="all_years",
        subfolder="monpol",
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
        frac: panel_df["avg_upstreamness"].quantile(1 - frac)
        for frac in tail_fracs
    }

    # For each tail, compute average share of aggregate effect
    avg_shares = {}

    for frac in tail_fracs:
        cutoff = cutoffs[frac]

        # restrict to firms in the top frac of upstreamness
        mask = panel_h["avg_upstreamness"] >= cutoff
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
    rows = []
    for j, p in enumerate(tails):
        rows.append(pd.DataFrame({
            "h": horizons,
            "percentile": p,
            "mean_pct": mean_pct[:, j],
            "ci_low_pct": lo_pct[:, j],
            "ci_high_pct": hi_pct[:, j],
        }))
    save_graph_data(
        output_path=output_path,
        file_name=f"contrib_percentiles_{var}_{country}_data",
        data=pd.concat(rows, ignore_index=True),
        year="all_years",
        subfolder="monpol",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

def contrib_shares_byclass_horizon_mean_of_ratios(
    panel_df: pd.DataFrame,
    panel_h: pd.DataFrame,
    var: str,
    output_path: str,
    country: str,
    class_col: str = "up_class",
    h_col: str = "h",
    firm_col: str = "vat",
    year_col: str = "year",
    turnover_col: str = "turnover",
    save: bool = True,
    drop_zero_denominator: bool = True,
):
    
    tot_col = f"tot_eff_{var}"
    if tot_col not in panel_h.columns:
        raise KeyError(f"panel_h must contain column '{tot_col}'")

    # sales shares by year 
    dfw = panel_df[[firm_col, year_col, turnover_col]].copy()
    dfw = dfw.sort_values([firm_col, year_col])
    dfw["total_turnover_year"] = dfw.groupby(year_col)[turnover_col].transform("sum")
    dfw["sales_share"] = dfw[turnover_col] / dfw["total_turnover_year"]

    # merge weights into panel_h
    ph = panel_h.merge(
        dfw[[firm_col, year_col, "sales_share"]],
        on=[firm_col, year_col],
        how="left",
        validate="m:1",
    ).copy()

    ph = ph.dropna(subset=[year_col, h_col, class_col, "sales_share", tot_col])

    # total aggregate effect by (year, h)
    agg_year = (
        ph.groupby([year_col, h_col])
          .apply(lambda g: np.nansum(g["sales_share"] * g[tot_col]))
          .rename("agg_eff")
          .reset_index()
    )

    # class contribution by (year, h, class)
    contrib_year = (
        ph.groupby([year_col, h_col, class_col])
          .apply(lambda g: np.nansum(g["sales_share"] * g[tot_col]))
          .rename("contrib")
          .reset_index()
    )

    share_long = contrib_year.merge(agg_year, on=[year_col, h_col], how="left")

    if drop_zero_denominator:
        share_long = share_long.dropna(subset=["agg_eff"])
        share_long = share_long[share_long["agg_eff"] != 0]

    share_long["share"] = share_long["contrib"] / share_long["agg_eff"]

    # average shares across years
    share_overall = (
        share_long.groupby([h_col, class_col])["share"]
                  .mean()
                  .rename("share_mean_years")
                  .reset_index()
    )

    share_wide = (
        share_overall.pivot(index=h_col, columns=class_col, values="share_mean_years")
                    .sort_index()
                    .sort_index(axis=1)
    )

    # save outputs
    if save:
        out_dir = os.path.join(output_path, "all_years", "monpol")
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, f"share_byclass_{var}_{country}.csv")
        share_wide.to_csv(csv_path)

    return share_wide, share_long


def contrib_shares_bypercentile_horizon_mean_of_ratios(
    panel_df: pd.DataFrame,
    panel_h: pd.DataFrame,
    var: str,
    output_path: str,
    country: str,
    percentiles=(0.10, 0.01, 0.0001, 0.00001),  # top 10%, 1%, 0.01%, 0.001%
    cutoff_mode: str = "yearly",  # "yearly" (recommended) or "global"
    firm_col: str = "vat",
    year_col: str = "year",
    h_col: str = "h",
    turnover_col: str = "turnover",
    upstream_col: str = "upstreamness",
    save: bool = True,
    drop_zero_denominator: bool = True,
):

    tot_col = f"tot_eff_{var}"
    if tot_col not in panel_h.columns:
        raise KeyError(f"panel_h must contain column '{tot_col}'")

    # sales shares by year
    dfw = panel_df[[firm_col, year_col, turnover_col, upstream_col]].copy()
    dfw = dfw.sort_values([firm_col, year_col])
    dfw["total_turnover_year"] = dfw.groupby(year_col)[turnover_col].transform("sum")
    dfw["sales_share"] = dfw[turnover_col] / dfw["total_turnover_year"]

    # merge weights and upstreamness into panel_h 
    ph = panel_h.merge(
        dfw[[firm_col, year_col, "sales_share", upstream_col]],
        on=[firm_col, year_col],
        how="left",
        validate="m:1",
        suffixes=("", "_df"),
    ).copy()

    ph = ph.dropna(subset=[year_col, h_col, "sales_share", tot_col, upstream_col])

    # total aggregate effect by (year, h)
    agg_year = (
        ph.groupby([year_col, h_col])
          .apply(lambda g: np.nansum(g["sales_share"] * g[tot_col]))
          .rename("agg_eff")
          .reset_index()
    )

    global_cutoffs = None
    if cutoff_mode == "global":
        global_cutoffs = {p: dfw[upstream_col].quantile(1 - p) for p in percentiles}
    elif cutoff_mode != "yearly":
        raise ValueError("cutoff_mode must be 'yearly' or 'global'")

    # compute shares for each percentile tail
    all_long = []

    for p in percentiles:
        if cutoff_mode == "global":
            cutoff_val = global_cutoffs[p]
            mask = ph[upstream_col] >= cutoff_val
            ph_tail = ph.loc[mask].copy()

            contrib_tail = (
                ph_tail.groupby([year_col, h_col])
                      .apply(lambda g: np.nansum(g["sales_share"] * g[tot_col]))
                      .rename("contrib")
                      .reset_index()
            )

        else:  # yearly cutoffs
            # compute cutoff within each year
            cutoff_by_year = (
                dfw.groupby(year_col)[upstream_col]
                   .quantile(1 - p)
                   .rename("cutoff")
                   .reset_index()
            )

            ph2 = ph.merge(cutoff_by_year, on=year_col, how="left")
            ph_tail = ph2[ph2[upstream_col] >= ph2["cutoff"]].copy()

            contrib_tail = (
                ph_tail.groupby([year_col, h_col])
                      .apply(lambda g: np.nansum(g["sales_share"] * g[tot_col]))
                      .rename("contrib")
                      .reset_index()
            )

        tmp = contrib_tail.merge(agg_year, on=[year_col, h_col], how="left")
        if drop_zero_denominator:
            tmp = tmp.dropna(subset=["agg_eff"])
            tmp = tmp[tmp["agg_eff"] != 0]

        tmp["share"] = tmp["contrib"] / tmp["agg_eff"]
        tmp["p"] = p
        all_long.append(tmp[[year_col, h_col, "p", "share"]])

    share_long = pd.concat(all_long, ignore_index=True)

    # average across years
    share_overall = (
        share_long.groupby([h_col, "p"])["share"]
                  .mean()
                  .rename("share_mean_years")
                  .reset_index()
    )

    # reshape to wide
    share_wide = (
        share_overall.pivot(index=h_col, columns="p", values="share_mean_years")
                     .sort_index()
                     .sort_index(axis=1)
    )

    # save
    if save:
        out_dir = os.path.join(output_path, "all_years", "monpol")
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, f"share_bypercentile_{var}_{country}.csv")

        share_wide.to_csv(csv_path)

    return share_wide, share_long

def plot_share_dots_by_class(
    share_wide,
    output_path=None,
    filename=None,
    xlabel="Quarter",
    ylabel="Contribution to total effect (%)",
    ncols=3,
    figsize=(18, 6),
    xtick_step=1,
    share_in_percent=True,
    ylims=None,
):
    
    # horizons and classes
    horizons = share_wide.index.to_numpy()
    classes = list(share_wide.columns)
    n_classes = len(classes)

    # layout
    nrows = int(np.ceil(n_classes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.array(axes).reshape(-1)

    # scale to percent if needed
    plot_df = share_wide.copy()
    if share_in_percent:
        plot_df = plot_df * 100.0

    # compute global y-limits if not provided
    if ylims is None:
        y_min = np.nanmin(plot_df.to_numpy())
        y_max = np.nanmax(plot_df.to_numpy())
        pad = 0.05 * (y_max - y_min) if (y_max > y_min) else 1.0
        ylims = (y_min - pad, y_max + pad)

    for j, cls in enumerate(classes):
        ax = axes[j]
        y = plot_df[cls].to_numpy()

        mask = ~np.isnan(y)
        x = horizons[mask]
        y = y[mask]

        ax.scatter(x, y, s=35)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"Upstreamness class {cls}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(horizons.min(), horizons.max() + 1, xtick_step))
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(*ylims)

    # remove unused axes
    for k in range(n_classes, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()

    # save if requested
    if output_path is not None:
        if filename is None:
            filename = "share_byclass_horizon.png"
        import os
        os.makedirs(output_path, exist_ok=True)
        fig_path = os.path.join(output_path, filename)
        save_graph_data(
            output_path=os.path.dirname(os.path.dirname(output_path)),
            file_name=f"{os.path.splitext(filename)[0]}_data",
            data=plot_df.reset_index().rename(columns={"index": "h"}),
            year="all_years",
            subfolder="monpol",
        )
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        
def contrib_shares_bybins_horizon_mean_of_ratios(
    panel_df: pd.DataFrame,
    panel_h: pd.DataFrame,
    var: str,
    output_path: str,
    country: str,
    bin_edges=(0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0),
    cutoff_mode: str = "yearly",   # "yearly" or "global"
    firm_col: str = "vat",
    year_col: str = "year",
    h_col: str = "h",
    turnover_col: str = "turnover",
    upstream_col: str = "avg_upstreamness",
    save: bool = True,
    drop_zero_denominator: bool = True,
):
    """
    Decompose the aggregate monetary policy response into contributions
    from percentile bins of the upstreamness distribution.

    Shares are computed as:

        share_{bin, year, h}
            = contribution_{bin, year, h} / aggregate_effect_{year, h}

    and then averaged across years.

    Bins are assigned at the firm-year level before merging into panel_h,
    so firm-year observations are not ranked repeatedly across horizons.

    Ties in upstreamness are broken deterministically using firm_col.
    Therefore bins contain approximately the intended percentile mass.
    """

    tot_col = f"tot_eff_{var}"

    if tot_col not in panel_h.columns:
        raise KeyError(f"panel_h must contain column '{tot_col}'")

    required_panel_df = [firm_col, year_col, turnover_col, upstream_col]
    missing_panel_df = [c for c in required_panel_df if c not in panel_df.columns]
    if missing_panel_df:
        raise KeyError(f"panel_df is missing columns: {missing_panel_df}")

    required_panel_h = [firm_col, year_col, h_col, tot_col]
    missing_panel_h = [c for c in required_panel_h if c not in panel_h.columns]
    if missing_panel_h:
        raise KeyError(f"panel_h is missing columns: {missing_panel_h}")

    # ------------------------------------------------------------
    # 1. Firm-year sales shares
    # ------------------------------------------------------------

    dfw = panel_df[[firm_col, year_col, turnover_col, upstream_col]].copy()

    dfw = dfw.dropna(
        subset=[firm_col, year_col, turnover_col, upstream_col]
    )

    dfw["total_turnover_year"] = (
        dfw.groupby(year_col)[turnover_col]
           .transform("sum")
    )

    dfw = dfw[dfw["total_turnover_year"] != 0].copy()

    dfw["sales_share"] = (
        dfw[turnover_col] / dfw["total_turnover_year"]
    )

    # ------------------------------------------------------------
    # 2. Percentile-bin assignment at firm-year level
    # ------------------------------------------------------------

    bin_edges_sorted = sorted(bin_edges)

    if bin_edges_sorted[0] != 0 or bin_edges_sorted[-1] != 1:
        raise ValueError("bin_edges should start at 0 and end at 1")

    if any((x < 0 or x > 1) for x in bin_edges_sorted):
        raise ValueError("bin_edges must lie between 0 and 1")

    if len(set(bin_edges_sorted)) != len(bin_edges_sorted):
        raise ValueError("bin_edges contains duplicate values")

    bin_labels = [
        f"p{int(bin_edges_sorted[i] * 100)}_p{int(bin_edges_sorted[i + 1] * 100)}"
        for i in range(len(bin_edges_sorted) - 1)
    ]

    if cutoff_mode == "yearly":
        dfw = dfw.sort_values([year_col, upstream_col, firm_col]).copy()

        dfw["_rank_num"] = dfw.groupby(year_col).cumcount() + 1
        dfw["_rank_den"] = dfw.groupby(year_col)[firm_col].transform("count")
        dfw["_pct_rank"] = dfw["_rank_num"] / dfw["_rank_den"]

    elif cutoff_mode == "global":
        dfw = dfw.sort_values([upstream_col, year_col, firm_col]).copy()
        dfw["_pct_rank"] = (np.arange(len(dfw)) + 1) / len(dfw)

    else:
        raise ValueError("cutoff_mode must be 'yearly' or 'global'")

    dfw["bin"] = pd.cut(
        dfw["_pct_rank"],
        bins=bin_edges_sorted,
        labels=bin_labels,
        include_lowest=True,
        right=True,
    )

    # ------------------------------------------------------------
    # 3. Merge bins and weights into horizon panel
    # ------------------------------------------------------------

    ph_base = panel_h.drop(
        columns=[
            c for c in ["sales_share", "_pct_rank", "bin"]
            if c in panel_h.columns
        ],
        errors="ignore",
    )

    ph = ph_base.merge(
        dfw[[firm_col, year_col, "sales_share", "_pct_rank", "bin"]],
        on=[firm_col, year_col],
        how="left",
        validate="m:1",
    ).copy()

    ph = ph.dropna(
        subset=[year_col, h_col, "sales_share", tot_col, "_pct_rank", "bin"]
    )

    ph["_weighted_eff"] = ph["sales_share"] * ph[tot_col]

    # ------------------------------------------------------------
    # 4. Aggregate effect by year and horizon
    # ------------------------------------------------------------

    agg_year = (
        ph.groupby([year_col, h_col])["_weighted_eff"]
          .sum()
          .reset_index()
          .rename(columns={"_weighted_eff": "agg_eff"})
    )

    # ------------------------------------------------------------
    # 5. Bin contribution by year, horizon, bin
    # ------------------------------------------------------------

    contrib = (
        ph.groupby([year_col, h_col, "bin"], observed=False)["_weighted_eff"]
          .sum()
          .reset_index()
          .rename(columns={"_weighted_eff": "contrib"})
    )

    # Full grid: empty bins are zero contributions, not missing shares.
    year_h = ph[[year_col, h_col]].drop_duplicates()

    full_grid = (
        year_h.assign(_key=1)
              .merge(
                  pd.DataFrame({"bin": bin_labels, "_key": 1}),
                  on="_key",
                  how="outer",
              )
              .drop(columns="_key")
    )

    contrib = (
        full_grid.merge(
            contrib,
            on=[year_col, h_col, "bin"],
            how="left",
        )
        .fillna({"contrib": 0.0})
    )

    share_long = contrib.merge(
        agg_year,
        on=[year_col, h_col],
        how="left",
    )

    if drop_zero_denominator:
        share_long = share_long.dropna(subset=["agg_eff"])
        share_long = share_long[share_long["agg_eff"] != 0].copy()

    share_long["share"] = share_long["contrib"] / share_long["agg_eff"]

    # ------------------------------------------------------------
    # 6. Average shares across years
    # ------------------------------------------------------------

    share_overall = (
        share_long.groupby([h_col, "bin"], observed=False)["share"]
                  .mean()
                  .reset_index()
                  .rename(columns={"share": "share_mean_years"})
    )

    share_wide = (
        share_overall.pivot(
            index=h_col,
            columns="bin",
            values="share_mean_years",
        )
        .reindex(columns=bin_labels)
        .sort_index()
    )

    # ------------------------------------------------------------
    # 7. Diagnostics
    # ------------------------------------------------------------

    bin_counts = (
        dfw.groupby([year_col, "bin"], observed=False)[firm_col]
           .count()
           .reset_index()
           .rename(columns={firm_col: "n_firms_bin"})
    )

    bin_counts_wide = (
        bin_counts.pivot(
            index=year_col,
            columns="bin",
            values="n_firms_bin",
        )
        .reindex(columns=bin_labels)
        .fillna(0)
        .astype(int)
    )

    diagnostics_base = (
        dfw.groupby(year_col)
           .agg(
               n_firms=(firm_col, "count"),
               min_pct=("_pct_rank", "min"),
               max_pct=("_pct_rank", "max"),
               n_unique_upstream=(upstream_col, "nunique"),
           )
    )

    diagnostics = diagnostics_base.join(bin_counts_wide)

    # ------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------

    if save:
        out_dir = os.path.join(output_path, "all_years", "monpol")
        os.makedirs(out_dir, exist_ok=True)

        share_wide.to_csv(
            os.path.join(out_dir, f"share_bybins_{var}_{country}.csv")
        )

        diagnostics.to_csv(
            os.path.join(out_dir, f"share_bybins_diagnostics_{var}_{country}.csv")
        )

    return share_wide, share_long, diagnostics

def master_monpol(panel_df, input_path, output_path, country):
    
    irfs_sales = pd.read_csv(os.path.join(input_path, 'irfs_int_avg_up_quarterly.csv')).rename(columns={'Unnamed: 0': 'h'})
    irfs_prices = pd.read_csv(os.path.join(input_path, 'irfs_p_int_avg_up_quarterly.csv')).rename(columns={'Unnamed: 0': 'h'})
    vcov_sales = pd.read_csv(os.path.join(input_path, 'vcov_int_avg_up_quarterly.csv'), index_col=0)
    vcov_prices = pd.read_csv(os.path.join(input_path, 'vcov_p_int_avg_up_quarterly.csv'), index_col=0)
    
    h = 12
    horizons = np.arange(h+1)

    # sort once
    panel_df = panel_df.sort_values(['vat', 'year']).reset_index(drop=True)
    #panel_df["avg_upstreamness"] = (
    #    panel_df.groupby("vat")["avg_upstreamness"].shift(1)
    #)
    panel_df["avg_upstreamness"] = panel_df["avg_upstreamness"].clip(upper=20) # cap upstreamness at 20

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
            irfs[['h','std_MP_median', 'std_MP_median_avg_upstreamness']].rename(
                columns={
                    'std_MP_median':f'beta_{var}',
                    'std_MP_median_avg_upstreamness': f'delta_{var}'
                }),
            on='h', how='left'
        )
        
        # delta-method firm-level total effects
        panel_h = add_tot_effect_and_se(panel_expanded, vcov, var, beta_col=f'beta_{var}', delta_col=f'delta_{var}')
        
        # upstreamness classes
        panel_h = panel_h.dropna(subset=['avg_upstreamness']).copy()
        panel_h["up_class"] = np.ceil(panel_h["avg_upstreamness"]).astype(int)
        panel_h.loc[panel_h["up_class"] > 5, "up_class"] = 5
        
        share_wide_pct, share_long_pct, diagnostics = contrib_shares_bybins_horizon_mean_of_ratios(
            panel_df, 
            panel_h, 
            var , 
            output_path, 
            country, 
            bin_edges=(0,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1))
        
        #plot_irfs(panel_h, output_path, fig_name, var=var)
        
        share_wide, share_long = contrib_shares_byclass_horizon_mean_of_ratios(
            panel_df=panel_df,
            panel_h=panel_h,
            var=var,
            output_path=output_path,
            country=country,
        )
        
        
        






        
        

        
