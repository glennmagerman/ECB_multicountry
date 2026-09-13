"""Render the repo figures from moments/key_moments.csv.

Usage (from the repo root):
    python figures/make_figures.py

Writes PNG files into figures/. All numbers come from the paper's tables via
moments/key_moments.csv, so the figures can be regenerated when the table changes.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CSV = os.path.join(ROOT, "moments", "key_moments.csv")

COUNTRIES = ["BE", "EE", "HU", "IT", "PT"]
NAMES = {"BE": "Belgium", "EE": "Estonia", "HU": "Hungary", "IT": "Italy", "PT": "Portugal"}
COLORS = {"BE": "#E69F00", "EE": "#CC79A7", "HU": "#56B4E9", "IT": "#0072B2", "PT": "#009E73"}  # Okabe-Ito
INK, MUTED, GRID, SURF, REST = "#1f2a37", "#6b7280", "#e5e7eb", "#ffffff", "#eef1f5"
SOURCE = "Source: Magerman et al. (2026), harmonised VAT transaction data across 5 EU countries."


def load_moments():
    with open(CSV, newline="") as f:
        return {r["moment"]: {c: float(r[c]) for c in COUNTRIES} for r in csv.DictReader(f)}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    })


def fig_concentration(m):
    """Top 1% and top 10% of supplier-customer relationships (annual bilateral flows m_ij), share of network sales (Table 2B)."""
    top1, top10 = m["top1_links_sales"], m["top10_links_sales"]
    order = sorted(COUNTRIES, key=lambda c: top1[c], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 6.2), dpi=200)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    y = list(range(5))[::-1]
    v1 = [top1[c] * 100 for c in order]
    v10 = [top10[c] * 100 for c in order]
    ax.barh(y, [100] * 5, height=0.62, color=REST, zorder=1)
    ax.barh(y, v10, height=0.62, color=[COLORS[c] for c in order], alpha=0.38, zorder=2)
    ax.barh(y, v1, height=0.62, color=[COLORS[c] for c in order], zorder=3)
    for yi, a, b in zip(y, v1, v10):
        ax.text(a - 2, yi, f"{a:.0f}%", va="center", ha="right", fontsize=16, color=SURF, fontweight="bold", zorder=4)
        ax.text(b - 2, yi, f"{b:.0f}%", va="center", ha="right", fontsize=13, color=INK, fontweight="bold", zorder=4)
    ax.text(0, 4.68, "top 1% of relationships", ha="left", va="bottom", fontsize=11, color=INK, fontweight="bold")
    ax.text(v10[0] - 1, 4.68, "top 10%", ha="right", va="bottom", fontsize=11, color=INK)
    ax.text(100, 4.68, "all", ha="right", va="bottom", fontsize=11, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels([NAMES[c] for c in order], fontsize=14)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=12)
    ax.set_ylim(-0.6, 5.3)
    ax.tick_params(axis="both", length=0); ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("Share of all sales between firms", fontsize=12.5, color=MUTED, labelpad=8)
    ax.text(0, 1.22, "Top 1% of supplier-customer relationships\naccount for up to 80% of network value", transform=ax.transAxes,
            fontsize=20, fontweight="bold", color=INK, va="bottom", linespacing=1.15)
    ax.text(0, 1.04, "The top 10% of relationships account for up to 96% of total network sales,\n"
                     "while the top 1% still represents between 50 and 80%.",
            transform=ax.transAxes, fontsize=12.5, color=MUTED, va="bottom", linespacing=1.3)
    fig.text(0.01, 0.01, SOURCE, fontsize=10, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    fig.savefig(os.path.join(HERE, "concentration_relationships.png"), facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    style()
    moments = load_moments()
    fig_concentration(moments)
    print("figures written to", HERE)
