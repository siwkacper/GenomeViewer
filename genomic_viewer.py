#!/usr/bin/env python3
"""
Genomic Data Viewer — interactive Dash + Plotly browser for genomic features.

Install:
    pip install dash dash-bootstrap-components plotly numpy pandas

Run:
    python genomic_viewer.py
    → open http://127.0.0.1:8050
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc


# ── Chromosome catalogue (GRCh37 sizes in bp) ─────────────────────────────
CHROMOSOMES = {
    "chr1":  249_250_621, "chr2":  243_199_373, "chr3":  198_022_430,
    "chr4":  191_154_276, "chr5":  180_915_260, "chr6":  171_115_067,
    "chr7":  159_138_663, "chr8":  146_364_022, "chr9":  141_213_431,
    "chr10": 135_534_747, "chr11": 135_006_516, "chr12": 133_851_895,
    "chr13": 115_169_878, "chr14": 107_349_540, "chr15": 102_531_392,
    "chr16":  90_354_753, "chr17":  81_195_210, "chr18":  78_077_248,
    "chr19":  59_128_983, "chr20":  63_025_520, "chr21":  48_129_895,
    "chr22":  51_304_566, "chrX":  155_270_560, "chrY":   59_373_566,
}


# ── Synthetic data generators ──────────────────────────────────────────────

def gen_genes(chrom: str, start: int, end: int, n: int = 20) -> pd.DataFrame:
    seed = abs(hash(f"{chrom}{start}")) % (2**31)
    rng = np.random.default_rng(seed)
    span = max(end - start, 1)
    offsets = np.sort(rng.integers(0, span, (n, 2)), axis=1)
    lengths = rng.integers(1_000, min(80_000, span // 2 + 1), n)
    biotypes = rng.choice(
        ["protein_coding", "lncRNA", "pseudogene"], n, p=[0.55, 0.30, 0.15]
    )
    return pd.DataFrame({
        "start":   start + offsets[:, 0],
        "end":     start + offsets[:, 0] + lengths,
        "name":    [f"GENE{i+1:03d}" for i in range(n)],
        "biotype": biotypes,
        "strand":  rng.choice(["+", "-"], n),
    })


def gen_coverage(start: int, end: int) -> pd.DataFrame:
    rng = np.random.default_rng(start % (2**31))
    n = min(2000, max(100, (end - start) // 100))
    pos = np.linspace(start, end, n, dtype=int)
    depth = rng.negative_binomial(30, 0.5, n).astype(float)
    gap_idx = rng.choice(n, n // 6, replace=False)
    depth[gap_idx] *= rng.uniform(0, 0.15, n // 6)
    return pd.DataFrame({"pos": pos, "depth": np.clip(depth, 0, None)})


def gen_variants(chrom: str, start: int, end: int, n: int = 100) -> pd.DataFrame:
    seed = abs(hash(f"{chrom}{end}")) % (2**31)
    rng = np.random.default_rng(seed)
    hi = max(start + 1, end)
    return pd.DataFrame({
        "pos":  np.sort(rng.integers(start, hi, n)),
        "type": rng.choice(["SNP", "INS", "DEL"], n, p=[0.70, 0.15, 0.15]),
        "af":   rng.beta(0.5, 6, n),
        "qual": rng.uniform(10, 60, n),
        "ref":  rng.choice(list("ACGT"), n),
    })


# ── Colour palettes ────────────────────────────────────────────────────────
BIOTYPE_CLR = {
    "protein_coding": "#1976D2",
    "lncRNA":         "#7B1FA2",
    "pseudogene":     "#2E7D32",
}
VARIANT_CLR = {"SNP": "#42A5F5", "INS": "#66BB6A", "DEL": "#EF5350"}


# ── Gene packing (greedy non-overlapping slots) ────────────────────────────

def assign_slots(genes: pd.DataFrame) -> tuple[dict, int]:
    slots: dict[str, int] = {}
    slot_ends: list[int] = []
    for _, g in genes.iterrows():
        placed = False
        for i, last_end in enumerate(slot_ends):
            if g["start"] > last_end + 300:
                slots[g["name"]] = i
                slot_ends[i] = int(g["end"])
                placed = True
                break
        if not placed:
            slots[g["name"]] = len(slot_ends)
            slot_ends.append(int(g["end"]))
    return slots, max(len(slot_ends), 1)


# ── Figure builder ─────────────────────────────────────────────────────────

def build_figure(
    chrom: str, start: int, end: int, tracks: list[str]
) -> go.Figure:
    active = [t for t in ("genes", "coverage", "variants") if t in tracks]
    if not active:
        active = ["coverage"]

    raw_heights = {"genes": 0.20, "coverage": 0.42, "variants": 0.38}
    row_h = [raw_heights[t] for t in active]
    total = sum(row_h)
    row_h = [h / total for h in row_h]

    fig = make_subplots(
        rows=len(active), cols=1,
        shared_xaxes=True,
        row_heights=row_h,
        vertical_spacing=0.05,
        subplot_titles=[t.capitalize() for t in active],
    )
    row = {t: i + 1 for i, t in enumerate(active)}

    # ── Gene track ──────────────────────────────────────────────────────
    if "genes" in tracks:
        genes = gen_genes(chrom, start, end)
        slots, n_slots = assign_slots(genes)
        for _, g in genes.iterrows():
            y = slots[g["name"]]
            c = BIOTYPE_CLR[g["biotype"]]
            arrow = "▶" if g["strand"] == "+" else "◀"
            hover = (
                f"<b>{g['name']}</b><br>"
                f"{g['biotype']}<br>"
                f"Strand: {g['strand']}<br>"
                f"Length: {g['end'] - g['start']:,} bp"
            )
            fig.add_trace(go.Scatter(
                x=[g["start"], g["end"], g["end"], g["start"], g["start"]],
                y=[y - 0.36, y - 0.36, y + 0.36, y + 0.36, y - 0.36],
                fill="toself", mode="lines",
                fillcolor=c,
                #fillcolor=c + "BB",
                line=dict(color=c, width=1),
                text=[hover] * 5, hoverinfo="text",
                name=g["name"], showlegend=False,
            ), row=row["genes"], col=1)
            fig.add_annotation(
                x=(g["start"] + g["end"]) / 2, y=y,
                text=f"{arrow} {g['name']}",
                showarrow=False,
                font=dict(size=8, color="white"),
                row=row["genes"], col=1,
            )
        fig.update_yaxes(
            showticklabels=False, showgrid=False,
            range=[-0.7, n_slots - 0.3],
            row=row["genes"], col=1,
        )

    # ── Coverage track ───────────────────────────────────────────────────
    if "coverage" in tracks:
        cov = gen_coverage(start, end)
        mean_d = cov["depth"].mean()
        fig.add_trace(go.Scatter(
            x=cov["pos"], y=cov["depth"],
            mode="lines", fill="tozeroy",
            line=dict(color="#42A5F5", width=0.8),
            fillcolor="rgba(66,165,245,0.20)",
            name="Depth",
            hovertemplate="pos %{x:,}<br>depth %{y:.0f}×<extra></extra>",
        ), row=row["coverage"], col=1)
        fig.add_hline(
            y=mean_d, line_dash="dot", line_color="#FFA726",
            annotation_text=f"mean {mean_d:.0f}×",
            annotation_font_color="#FFA726",
            row=row["coverage"], col=1,
        )
        fig.update_yaxes(
            title_text="Depth (×)", row=row["coverage"], col=1
        )

    # ── Variant track ────────────────────────────────────────────────────
    if "variants" in tracks:
        variants = gen_variants(chrom, start, end)
        for vtype, grp in variants.groupby("type"):
            fig.add_trace(go.Scatter(
                x=grp["pos"],
                y=grp["af"],
                mode="markers",
                marker=dict(
                    color=VARIANT_CLR[vtype],
                    size=4 + grp["qual"] / 10,
                    opacity=0.78,
                    line=dict(width=0.4, color="rgba(255,255,255,0.2)"),
                ),
                name=vtype,
                customdata=np.stack(
                    [grp["ref"], grp["qual"].round(1)], axis=1
                ),
                hovertemplate=(
                    f"<b>{vtype}</b><br>"
                    "pos %{x:,}<br>"
                    "AF %{y:.3f}<br>"
                    "ref %{customdata[0]}<br>"
                    "QUAL %{customdata[1]}<extra></extra>"
                ),
            ), row=row["variants"], col=1)
        fig.update_yaxes(
            title_text="Allele freq.", range=[-0.05, 1.05],
            row=row["variants"], col=1,
        )

    # ── Shared x-axis label ──────────────────────────────────────────────
    fig.update_xaxes(
        title_text=f"{chrom} coordinate",
        tickformat=",",
        range=[start, end],
        row=len(active), col=1,
    )
    fig.update_layout(
        height=690,
        template="plotly_dark",
        paper_bgcolor="#12131f",
        plot_bgcolor="#12131f",
        font=dict(family="monospace", size=11, color="#dde1f0"),
        legend=dict(orientation="h", y=-0.12, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=65, r=20, t=50, b=65),
        hovermode="x",
    )
    return fig


# ── Dash application ───────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Genomic Viewer"

app.layout = dbc.Container([
    html.H4(
        "Genomic Data Viewer",
        className="text-center my-3 text-info fw-bold",
    ),

    dbc.Card(dbc.CardBody(
        dbc.Row([
            # Chromosome selector
            dbc.Col([
                dbc.Label("Chromosome", className="small fw-semibold"),
                dcc.Dropdown(
                    id="chrom",
                    options=[{"label": c, "value": c} for c in CHROMOSOMES],
                    value="chr1",
                    clearable=False,
                    className="text-dark",
                ),
            ], xs=12, sm=4, md=2),

            # Start coordinate
            dbc.Col([
                dbc.Label("Start (bp)", className="small fw-semibold"),
                dbc.Input(
                    id="reg-start", type="number",
                    value=10_000_000, step=10_000, min=1,
                ),
            ], xs=6, sm=4, md=2),

            # End coordinate
            dbc.Col([
                dbc.Label("End (bp)", className="small fw-semibold"),
                dbc.Input(
                    id="reg-end", type="number",
                    value=11_000_000, step=10_000, min=1,
                ),
            ], xs=6, sm=4, md=2),

            # Track toggles
            dbc.Col([
                dbc.Label("Tracks", className="small fw-semibold"),
                dbc.Checklist(
                    id="tracks",
                    options=[
                        {"label": " Genes",    "value": "genes"},
                        {"label": " Coverage", "value": "coverage"},
                        {"label": " Variants", "value": "variants"},
                    ],
                    value=["genes", "coverage", "variants"],
                    inline=True, switch=True,
                ),
            ], xs=12, sm=8, md=4),

            # Load button
            dbc.Col([
                html.Br(),
                dbc.Button(
                    "Load region", id="go",
                    color="primary", n_clicks=0, className="w-100",
                ),
            ], xs=12, sm=4, md=2, className="d-flex align-items-end"),
        ], className="g-2 align-items-end"),
    ), className="mb-3"),

    dcc.Graph(
        id="graph",
        config={"scrollZoom": True, "displayModeBar": True},
    ),
    html.Small(
        id="region-info",
        className="text-muted d-block text-center mt-1",
    ),
], fluid=True)


@app.callback(
    Output("graph", "figure"),
    Output("region-info", "children"),
    Input("go", "n_clicks"),
    State("chrom", "value"),
    State("reg-start", "value"),
    State("reg-end", "value"),
    State("tracks", "value"),
)
def render(_, chrom, start, end, tracks):
    chrom = chrom or "chr1"
    start = max(1, int(start or 10_000_000))
    end = int(end or 11_000_000)
    chrom_len = CHROMOSOMES.get(chrom, 250_000_000)
    end = min(end, chrom_len)
    if end <= start:
        end = start + 500_000

    fig = build_figure(chrom, start, end, tracks or ["coverage"])
    info = (
        f"{chrom}:{start:,}–{end:,}"
        f"  ·  region {end - start:,} bp"
        f"  ·  chromosome {chrom_len:,} bp"
    )
    return fig, info


if __name__ == "__main__":
    app.run(debug=True, port=8050)
