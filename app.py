"""
Credit Risk & Early Warning System — portfolio risk dashboard.

Every panel is backed by SQL (analytics views / tables in PostgreSQL or SQLite).
Run:  streamlit run app.py
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src.db import get_engine, read_sql
from src.stress import MACRO_VARS, SCENARIOS, apply_scenario

st.set_page_config(page_title="Credit Risk & EWS", page_icon="🏦", layout="wide")

# ----------------------------------------------------------------------------- palette
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PRODUCTS = ["Home Loan", "Auto Loan", "Personal Loan", "Credit Card", "Gold Loan", "Business Loan"]
PRODUCT_COLOR = dict(zip(PRODUCTS, SERIES))
GOOD, WARN, SERIOUS, CRIT = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
STAGE_COLOR = {1: GOOD, 2: WARN, 3: CRIT}
EWS_COLOR = {"Green": GOOD, "Amber": WARN, "Red": CRIT}
SEVERITY = ["Low", "Medium", "High", "Critical"]
SEV_COLOR = dict(zip(SEVERITY, [GOOD, WARN, SERIOUS, CRIT]))
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SCEN_COLOR = {"Baseline": "#86b6ef", "Adverse": "#2a78d6", "Severe": "#104281", "Custom": "#eb6834"}
RISK_UP, RISK_DOWN = "#e34948", "#2a78d6"

st.markdown(f"""
<style>
  .block-container {{padding-top: 1.6rem; max-width: 1400px;}}
  [data-testid="stMetric"] {{background: {SURFACE}; border: 1px solid rgba(11,11,11,0.10);
      border-radius: 10px; padding: 12px 16px;}}
  [data-testid="stMetricLabel"] p {{color: {INK2}; font-size: 0.82rem;}}
  [data-testid="stMetricValue"] {{font-size: 1.55rem;}}
  .pill {{display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem;
      font-weight:600; margin-right:6px;}}
  .muted {{color:{INK2}; font-size:0.86rem;}}
  h3 {{margin-top: 0.4rem;}}
</style>""", unsafe_allow_html=True)


def pill(text, color, dark_text=False):
    fg = INK if dark_text else "#ffffff"
    return f'<span class="pill" style="background:{color};color:{fg}">{text}</span>'


def status_pill(level):
    icon = {"Green": "●", "Amber": "▲", "Red": "■", "Low": "●", "Medium": "▲", "High": "◆", "Critical": "■"}
    color = {**EWS_COLOR, **SEV_COLOR}.get(level, MUTED)
    return pill(f"{icon.get(level, '')} {level}", color, dark_text=level in ("Amber", "Medium"))


def style(fig, height=340, legend=True, xtitle=None, ytitle=None):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE, font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                                        color=INK2, size=12),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None),
        barcornerradius=4, bargap=0.28, hoverlabel=dict(bgcolor="white", font_color=INK),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE, zeroline=False, title=xtitle,
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, zeroline=False, title=ytitle,
                     tickfont=dict(color=MUTED))
    return fig


def cr(x):
    return f"₹{x / 1e7:,.1f} cr"


def lakh(x):
    return f"₹{x / 1e5:,.1f} L"


# ----------------------------------------------------------------------------- data access
@st.cache_resource
def engine():
    return get_engine()


@st.cache_data(ttl=600)
def q(sql, **params):
    return read_sql(sql, engine(), params or None)


def load_json(name):
    p = config.ARTIFACTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else {}


try:
    kpi = q("SELECT * FROM v_portfolio_kpis").iloc[0]
except Exception as exc:  # pipeline not run yet
    st.error("No analytics tables found. Run `python run_pipeline.py` first.")
    st.caption(str(exc)[:300])
    st.stop()

report = load_json("model_report.json")
summary = load_json("run_summary.json")

# ----------------------------------------------------------------------------- header
st.markdown("## 🏦 Credit Risk & Early Warning System")
st.markdown(f'<div class="muted">Retail lending portfolio · snapshot {config.SNAPSHOT_DATE} · '
            f'{int(kpi.n_customers):,} borrowers · {int(kpi.n_loans):,} facilities · '
            f'database: <b>{engine().dialect.name}</b></div>', unsafe_allow_html=True)
st.write("")

c = st.columns(6)
c[0].metric("Exposure at Default", cr(kpi.total_ead))
c[1].metric("Expected Credit Loss", cr(kpi.total_ecl), f"{kpi.coverage_ratio:.2%} coverage", delta_color="off", delta_arrow="off")
c[2].metric("Gross NPA (Stage 3)", f"{kpi.gnpa_ratio:.2%}", "of EAD", delta_color="off", delta_arrow="off")
c[3].metric("Stage 2 (SICR)", f"{kpi.stage2_ratio:.2%}", "of EAD", delta_color="off", delta_arrow="off")
c[4].metric("EWS watch-list", f"{int(kpi.n_ews_red):,} Red", f"{int(kpi.n_ews_amber):,} Amber", delta_color="off", delta_arrow="off")
c[5].metric("AML alerts", f"{int(kpi.n_aml_flagged):,}", "accounts flagged", delta_color="off", delta_arrow="off")

tabs = st.tabs(["Portfolio", "Risk distribution", "Early warning", "Account drill-down",
                "Anomaly detection", "Stress testing", "Model performance"])

# ============================================================================= 1. PORTFOLIO
with tabs[0]:
    prod = q("SELECT * FROM v_exposure_by_product")
    prod["product"] = pd.Categorical(prod["product"], PRODUCTS, ordered=True)
    prod = prod.sort_values("product")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("### Exposure by product and IFRS 9 stage")
        fig = go.Figure()
        for s, col, name in [(1, "stage1_ead", "Stage 1 · performing"), (2, "stage2_ead", "Stage 2 · SICR"),
                             (3, "stage3_ead", "Stage 3 · credit-impaired")]:
            fig.add_bar(y=prod["product"].astype(str), x=prod[col] / 1e7, orientation="h", name=name,
                        marker=dict(color=STAGE_COLOR[s], line=dict(color=SURFACE, width=2)),
                        hovertemplate="%{y}<br>" + name + ": ₹%{x:,.1f} cr<extra></extra>")
        fig.update_layout(barmode="stack", legend_traceorder="normal")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 330, xtitle="EAD (₹ crore)"), width="stretch")
    with right:
        st.markdown("### ECL and coverage by product")
        fig = go.Figure(go.Bar(
            x=prod["product"].astype(str), y=prod["ecl"] / 1e7,
            marker_color=[PRODUCT_COLOR[p] for p in prod["product"].astype(str)],
            text=[f"{v:.1%}" for v in prod["coverage"]], textposition="outside", textfont=dict(color=INK2),
            customdata=np.c_[prod["wavg_pd"], prod["wavg_lgd"], prod["npa_ratio"]],
            hovertemplate="%{x}<br>ECL ₹%{y:,.2f} cr<br>Coverage %{text}<br>EAD-wtd PD %{customdata[0]:.2%}"
                          "<br>EAD-wtd LGD %{customdata[1]:.1%}<br>NPA %{customdata[2]:.2%}<extra></extra>"))
        st.plotly_chart(style(fig, 330, legend=False, ytitle="ECL (₹ crore)"), width="stretch")
        st.caption("Label = ECL / EAD coverage. Unsecured products carry higher LGD and coverage.")

    st.markdown("### Concentration")
    l2, r2 = st.columns([1.15, 1])
    with l2:
        dim = st.radio("Segment by", ["State", "Employment", "City tier"], horizontal=True,
                       label_visibility="collapsed")
        seg = q("SELECT * FROM v_exposure_by_segment WHERE dimension = :d", d=dim).sort_values("ead")
        fig = go.Figure(go.Bar(
            y=seg["segment"], x=seg["ead"] / 1e7, orientation="h", marker_color=SERIES[0],
            customdata=np.c_[seg["wavg_pd"], seg["coverage"], seg["n_loans"]],
            hovertemplate="%{y}<br>EAD ₹%{x:,.1f} cr<br>EAD-wtd PD %{customdata[0]:.2%}"
                          "<br>Coverage %{customdata[1]:.2%}<br>%{customdata[2]:,} loans<extra></extra>"))
        st.plotly_chart(style(fig, 360, legend=False, xtitle="EAD (₹ crore)"), width="stretch")
        hhi = ((seg["ead"] / seg["ead"].sum()) ** 2).sum()
        st.caption(f"Herfindahl–Hirschman index across {dim.lower()} segments: {hhi:.3f}")
    with r2:
        ts = q("SELECT * FROM v_ticket_size ORDER BY bucket_order")
        fig = go.Figure(go.Bar(x=ts["bucket"], y=ts["ead"] / 1e7, marker_color=SERIES[0],
                               customdata=ts["n_loans"],
                               hovertemplate="%{x}<br>EAD ₹%{y:,.1f} cr<br>%{customdata:,} loans<extra></extra>"))
        fig.update_layout(title=dict(text="Exposure by ticket size", font=dict(size=13, color=INK)))
        st.plotly_chart(style(fig, 360, legend=False, ytitle="EAD (₹ crore)"), width="stretch")

    st.markdown("### Top 25 borrower exposures")
    top = q("SELECT * FROM v_top_exposures")
    top_view = top.assign(total_ead=top["total_ead"].map(lakh), total_ecl=top["total_ecl"].map(lakh),
                          pd_12m=top["pd_12m"].map("{:.2%}".format))
    st.dataframe(top_view, hide_index=True, width="stretch", height=300)

# ============================================================================= 2. RISK DISTRIBUTION
with tabs[1]:
    cr_df = q("SELECT customer_id, pd_12m, risk_score, risk_band, risk_severity, ews_level, sma_class, "
              "total_ead, total_ecl, max_stage FROM customer_risk")
    sev_counts = cr_df["risk_severity"].value_counts()
    cols = st.columns(4)
    for col, s in zip(cols, SEVERITY):
        sub = cr_df[cr_df["risk_severity"] == s]
        col.markdown(status_pill(s), unsafe_allow_html=True)
        col.metric(f"{s} severity", f"{len(sub):,} accounts", f"{cr(sub['total_ead'].sum())} EAD",
                   delta_color="off", delta_arrow="off", label_visibility="collapsed")

    a, b = st.columns(2)
    with a:
        st.markdown("### 12-month PD distribution (performing book)")
        perf = cr_df[cr_df["max_stage"] < 3]
        fig = go.Figure(go.Histogram(x=np.log10(perf["pd_12m"].clip(1e-4)), nbinsx=50, marker_color=SERIES[0],
                                     marker_line=dict(color=SURFACE, width=1),
                                     hovertemplate="log10 PD %{x:.2f}<br>%{y:,} accounts<extra></extra>"))
        ticks = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
        fig.update_xaxes(tickvals=np.log10(ticks), ticktext=[f"{t:.1%}" for t in ticks])
        st.plotly_chart(style(fig, 330, legend=False, xtitle="PD (log scale)", ytitle="Accounts"),
                        width="stretch")
    with b:
        st.markdown("### Risk bands (PD grade)")
        rb = q("SELECT * FROM v_risk_band_summary ORDER BY risk_band")
        fig = go.Figure(go.Bar(
            x=rb["risk_band"], y=rb["ead"] / 1e7, marker_color=BLUES[1:][:len(rb)],
            customdata=np.c_[rb["n_customers"], rb["avg_pd"], rb["min_score"], rb["max_score"], rb["ecl"] / 1e7],
            hovertemplate="%{x}<br>EAD ₹%{y:,.1f} cr<br>%{customdata[0]:,} borrowers<br>Avg PD %{customdata[1]:.2%}"
                          "<br>Score %{customdata[2]}–%{customdata[3]}<br>ECL ₹%{customdata[4]:,.2f} cr<extra></extra>"))
        st.plotly_chart(style(fig, 330, legend=False, ytitle="EAD (₹ crore)"), width="stretch")

    a, b = st.columns([1.2, 1])
    with a:
        st.markdown("### Severity × product (EAD, ₹ crore)")
        sm = q("SELECT * FROM v_severity_matrix")
        piv = sm.pivot_table(index="risk_severity", columns="product", values="ead", aggfunc="sum") \
                .reindex(index=SEVERITY[::-1], columns=PRODUCTS).fillna(0) / 1e7
        fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index,
                                   colorscale=[[i / 6, c] for i, c in enumerate(BLUES)],
                                   text=np.round(piv.values, 1), texttemplate="%{text}",
                                   hovertemplate="%{y} · %{x}<br>EAD ₹%{z:,.1f} cr<extra></extra>",
                                   colorbar=dict(title="₹ cr", thickness=10)))
        fig.update_traces(xgap=2, ygap=2)
        st.plotly_chart(style(fig, 300, legend=False), width="stretch")
    with b:
        st.markdown("### RBI SMA classification")
        sma = q("SELECT * FROM v_sma_summary")
        order = ["Standard", "SMA-0", "SMA-1", "SMA-2", "NPA"]
        sma = sma.set_index("sma_class").reindex(order).fillna(0).reset_index()
        colors = [GOOD, BLUES[3], WARN, SERIOUS, CRIT]
        fig = go.Figure(go.Bar(x=sma["sma_class"], y=sma["n_customers"], marker_color=colors,
                               text=sma["n_customers"].astype(int), textposition="outside",
                               customdata=np.c_[sma["ead"] / 1e7, sma["avg_pd"]],
                               hovertemplate="%{x}<br>%{y:,} borrowers<br>EAD ₹%{customdata[0]:,.1f} cr"
                                             "<br>Avg PD %{customdata[1]:.2%}<extra></extra>"))
        fig.update_yaxes(type="log")
        st.plotly_chart(style(fig, 300, legend=False, ytitle="Borrowers (log)"), width="stretch")
        st.caption("SMA-0: 1–30 DPD · SMA-1: 31–60 · SMA-2: 61–90 · NPA: 90+")

# ============================================================================= 3. EARLY WARNING
with tabs[2]:
    freq = q("SELECT * FROM v_ews_signal_frequency").sort_values("n_customers")
    cats = ["Repayment", "Account conduct", "Leverage", "Bureau"]
    a, b = st.columns([1.3, 1])
    with a:
        st.markdown("### Signal frequency")
        fig = go.Figure()
        for i, cat in enumerate(cats):
            f = freq[freq["category"] == cat]
            fig.add_bar(y=f["signal_code"] + " · " + f["signal_name"], x=f["n_customers"], orientation="h",
                        name=cat, marker_color=SERIES[i], customdata=np.c_[f["weight"], f["avg_pd"]],
                        hovertemplate="%{y}<br>%{x:,} borrowers<br>Weight %{customdata[0]}"
                                      "<br>Avg PD of flagged %{customdata[1]:.2%}<extra></extra>")
        fig.update_layout(barmode="stack", yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(style(fig, 470, xtitle="Borrowers triggering"), width="stretch")
    with b:
        st.markdown("### Does the EWS agree with the PD model?")
        lv = q("SELECT ews_level, COUNT(*) AS n, AVG(pd_12m) AS avg_pd, SUM(total_ead) AS ead "
               "FROM customer_risk WHERE max_stage < 3 GROUP BY ews_level")
        lv = lv.set_index("ews_level").reindex(["Green", "Amber", "Red"]).reset_index()
        fig = go.Figure(go.Bar(x=lv["ews_level"], y=lv["avg_pd"], marker_color=[EWS_COLOR[x] for x in lv["ews_level"]],
                               text=[f"{v:.1%}" for v in lv["avg_pd"]], textposition="outside",
                               customdata=np.c_[lv["n"], lv["ead"] / 1e7],
                               hovertemplate="%{x}<br>Avg PD %{y:.2%}<br>%{customdata[0]:,} borrowers"
                                             "<br>EAD ₹%{customdata[1]:,.1f} cr<extra></extra>"))
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style(fig, 300, legend=False, ytitle="Average 12m PD"), width="stretch")
        dr = summary.get("ews_default_rate_by_level", {})
        if dr:
            st.markdown(f'<div class="muted">Back-test on realised outcomes (performing book): 12-month default '
                        f'rate <b>{dr.get("Green", 0):.1%}</b> Green → <b>{dr.get("Amber", 0):.1%}</b> Amber → '
                        f'<b>{dr.get("Red", 0):.1%}</b> Red.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="muted" style="margin-top:8px">Watch-list rule: weighted signal score ≥ 8 or '
                    f'SMA-2/NPA → Red; ≥ 4 or SMA-1 → Amber.</div>', unsafe_allow_html=True)

    st.markdown("### Watch-list")
    wl = q("SELECT * FROM v_ews_watchlist")
    sig = q("SELECT customer_id, signal_code FROM ews_signals")
    sig_txt = sig.groupby("customer_id")["signal_code"].apply(lambda s: ", ".join(sorted(s)))
    wl["signals"] = wl["customer_id"].map(sig_txt).fillna("")
    f1, f2, f3 = st.columns([1, 1, 2])
    lvl = f1.multiselect("EWS level", ["Red", "Amber"], default=["Red", "Amber"])
    stg = f2.multiselect("Stage", [1, 2, 3], default=[1, 2])
    srch = f3.text_input("Search customer or reason", "")
    view = wl[wl["ews_level"].isin(lvl) & wl["max_stage"].isin(stg)]
    if srch:
        m = view["customer_id"].str.contains(srch, case=False) | view["top_risk_driver"].str.contains(srch, case=False)
        view = view[m]
    view = view.sort_values(["ews_score", "pd_12m"], ascending=False)
    st.dataframe(view[["customer_id", "ews_level", "ews_score", "signals", "sma_class", "current_dpd", "pd_12m",
                       "risk_band", "max_stage", "total_ead", "total_ecl", "top_risk_driver"]],
                 hide_index=True, width="stretch", height=380,
                 column_config={"pd_12m": st.column_config.ProgressColumn("PD 12m", format="percent", min_value=0,
                                                                          max_value=1),
                                "total_ead": st.column_config.NumberColumn("EAD (₹)", format="%,.0f"),
                                "total_ecl": st.column_config.NumberColumn("ECL (₹)", format="%,.0f"),
                                "max_stage": "Stage"})
    st.caption(f"{len(view):,} accounts shown. Pick one in the Account drill-down tab for full explanations.")

# ============================================================================= 4. ACCOUNT DRILL-DOWN
with tabs[3]:
    ids = q("SELECT customer_id FROM customer_risk "
            "ORDER BY CASE WHEN max_stage = 3 THEN 1 ELSE 0 END, ews_score DESC, pd_12m DESC")["customer_id"].tolist()
    cid = st.selectbox("Customer", ids, index=0, help="Performing accounts first, sorted by EWS score then PD")
    r = q("SELECT * FROM customer_risk WHERE customer_id = :c", c=cid).iloc[0]
    prof = q("SELECT * FROM customers WHERE customer_id = :c", c=cid).iloc[0]

    st.markdown(
        f"{status_pill(r.risk_severity)} {pill('EWS ' + r.ews_level, EWS_COLOR[r.ews_level], r.ews_level == 'Amber')}"
        f"{pill(r.sma_class, MUTED)}{pill('Stage ' + str(int(r.max_stage)), STAGE_COLOR[int(r.max_stage)], r.max_stage == 2)}"
        + (pill('AML ' + r.alert_priority, CRIT) if r.is_flagged else ""), unsafe_allow_html=True)
    k = st.columns(6)
    k[0].metric("12m PD", f"{r.pd_12m:.2%}")
    k[1].metric("Risk score", f"{int(r.risk_score)}", r.risk_band, delta_color="off", delta_arrow="off")
    k[2].metric("EAD", lakh(r.total_ead))
    k[3].metric("ECL", lakh(r.total_ecl))
    k[4].metric("Monthly income", f"₹{prof.monthly_income:,.0f}")
    k[5].metric("Bureau score", f"{prof.bureau_score}", f"{prof.bureau_score - prof.bureau_score_6m_ago:+d} in 6m")
    st.markdown(f'<div class="muted">{prof.employment_type} · age {prof.age} · {prof.state} (Tier {prof.city_tier}) · '
                f'{prof.years_with_bank} yrs with bank · KYC risk {prof.kyc_risk}</div>', unsafe_allow_html=True)

    a, b = st.columns([1.1, 1])
    with a:
        st.markdown("### Why this PD? — SHAP risk drivers")
        drv = q("SELECT * FROM risk_drivers WHERE customer_id = :c ORDER BY driver_rank", c=cid)
        if drv.empty:
            st.info("Account is already NPA (90+ DPD): PD = 100%, no model explanation needed.")
        else:
            d = drv.sort_values("shap_value")
            fig = go.Figure(go.Bar(
                y=d["reason_text"], x=d["shap_value"], orientation="h",
                marker_color=np.where(d["shap_value"] > 0, RISK_UP, RISK_DOWN),
                customdata=np.c_[d["feature"], d["reason_code"]],
                hovertemplate="%{y}<br>%{customdata[1]} · %{customdata[0]}<br>SHAP %{x:+.3f} log-odds<extra></extra>"))
            st.plotly_chart(style(fig, 330, legend=False, xtitle="Contribution to log-odds of default"),
                            width="stretch")
            st.markdown("**Reason codes:** " + " · ".join(
                f"`{x.reason_code}` {x.reason_text}" for x in drv[drv["shap_value"] > 0].head(4).itertuples()))
    with b:
        st.markdown("### Early warning signals")
        s = q("SELECT signal_code, signal_name, category, weight FROM ews_signals WHERE customer_id = :c", c=cid)
        if s.empty:
            st.success("No early warning signals triggered.")
        else:
            st.dataframe(s, hide_index=True, width="stretch")
        h = q("SELECT rule_code, rule_name FROM aml_rule_hits WHERE customer_id = :c", c=cid)
        if not h.empty:
            st.markdown("**AML typology hits**")
            st.dataframe(h, hide_index=True, width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown("### Repayment history (DPD by facility)")
        ph = q("""SELECT p.loan_id, l.product, p.month_idx, p.due_date, p.dpd, p.due_amount, p.paid_amount
                  FROM payments p JOIN loans l ON l.loan_id = p.loan_id
                  WHERE l.customer_id = :c ORDER BY p.month_idx DESC""", c=cid)
        fig = go.Figure()
        for lid, g in ph.groupby("loan_id"):
            prod_name = g["product"].iloc[0]
            fig.add_scatter(x=g["due_date"], y=g["dpd"], mode="lines+markers", name=f"{prod_name} ({lid})",
                            line=dict(color=PRODUCT_COLOR[prod_name], width=2), marker=dict(size=8),
                            customdata=np.c_[g["due_amount"], g["paid_amount"]],
                            hovertemplate="%{x}<br>DPD %{y}<br>Due ₹%{customdata[0]:,.0f} · Paid ₹%{customdata[1]:,.0f}"
                                          "<extra>" + prod_name + "</extra>")
        for y, lab in [(30, "SMA-1"), (60, "SMA-2"), (90, "NPA")]:
            fig.add_hline(y=y, line=dict(color=BASELINE, dash="dot", width=1),
                          annotation_text=lab, annotation_font_color=MUTED, annotation_position="top left")
        st.plotly_chart(style(fig, 320, ytitle="Days past due"), width="stretch")
    with b:
        st.markdown("### Account cash-flows (last 6 months)")
        fl = q("""SELECT month_idx, MIN(txn_date) AS month,
                         SUM(CASE WHEN direction='CR' AND status='SUCCESS' THEN amount ELSE 0 END) AS inflow,
                         SUM(CASE WHEN direction='DR' AND status='SUCCESS' THEN amount ELSE 0 END) AS outflow,
                         SUM(CASE WHEN status='RETURNED' THEN 1 ELSE 0 END) AS bounces
                  FROM transactions WHERE customer_id = :c GROUP BY month_idx ORDER BY month_idx DESC""", c=cid)
        fl["month"] = pd.to_datetime(fl["month"]).dt.strftime("%b %y")
        fig = go.Figure()
        fig.add_bar(x=fl["month"], y=fl["inflow"], name="Inflows", marker_color=SERIES[0])
        fig.add_bar(x=fl["month"], y=fl["outflow"], name="Outflows", marker_color=SERIES[1])
        for _, row in fl[fl["bounces"] > 0].iterrows():
            fig.add_annotation(x=row["month"], y=max(row["inflow"], row["outflow"]), text=f"{int(row['bounces'])} bounce",
                               showarrow=False, yshift=12, font=dict(color=CRIT, size=11))
        fig.update_layout(barmode="group", bargroupgap=0.08)
        fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
        st.plotly_chart(style(fig, 320), width="stretch")

    st.markdown("### Facilities and expected credit loss")
    ln = q("SELECT * FROM ecl_results WHERE customer_id = :c", c=cid)
    st.dataframe(ln[["loan_id", "product", "secured", "outstanding", "collateral_value", "loan_dpd", "stage",
                     "pd_12m", "lifetime_pd", "lgd", "ead", "ecl", "coverage"]],
                 hide_index=True, width="stretch",
                 column_config={c_: st.column_config.NumberColumn(format="₹%,.0f")
                                for c_ in ["outstanding", "collateral_value", "ead", "ecl"]}
                 | {c_: st.column_config.NumberColumn(format="percent")
                    for c_ in ["pd_12m", "lifetime_pd", "lgd", "coverage"]})

# ============================================================================= 5. ANOMALY DETECTION
with tabs[4]:
    am = summary.get("anomaly_metrics", {})
    alerts = q("SELECT * FROM v_anomaly_alerts ORDER BY alert_score DESC")
    k = st.columns(5)
    k[0].metric("Accounts flagged", f"{len(alerts):,}")
    k[1].metric("High priority", f"{(alerts['alert_priority'] == 'High').sum():,}")
    k[2].metric("Transaction alerts", f"{int(alerts['n_txn_alerts'].sum()):,}")
    if am:
        k[3].metric("Precision / recall*", f"{am['precision']:.0%} / {am['recall']:.0%}")
        k[4].metric("Isolation Forest AUC*", f"{am['isolation_forest_auc']:.3f}")
        st.caption("*Measured against injected synthetic AML typologies (structuring, pass-through, odd-hour wires, "
                   "dormant bursts). Legitimate look-alikes — FD maturities, cash-heavy businesses, overseas "
                   "remittances — are deliberately present to create realistic false positives.")

    a, b = st.columns([1, 1.2])
    with a:
        st.markdown("### Typology rule hits")
        rs = q("SELECT * FROM v_aml_rule_summary ORDER BY n_customers")
        fig = go.Figure(go.Bar(y=rs["rule_code"] + " · " + rs["rule_name"], x=rs["n_customers"], orientation="h",
                               marker_color=SERIES[6], customdata=rs["weight"],
                               hovertemplate="%{y}<br>%{x:,} accounts<br>Weight %{customdata}<extra></extra>"))
        st.plotly_chart(style(fig, 340, legend=False, xtitle="Accounts"), width="stretch")
    with b:
        st.markdown("### Isolation Forest score vs inflow-to-income")
        sc = q("SELECT customer_id, isolation_percentile, inflow_to_income, is_flagged, alert_score FROM anomaly_scores")
        fig = go.Figure()
        for flag, name, col, size in [(0, "Not flagged", "#c3c2b7", 5), (1, "Flagged", CRIT, 8)]:
            s_ = sc[sc["is_flagged"] == flag]
            fig.add_scattergl(x=s_["isolation_percentile"], y=s_["inflow_to_income"].clip(upper=30), mode="markers",
                              name=name, marker=dict(color=col, size=size, opacity=0.7 if flag else 0.35,
                                                     line=dict(color=SURFACE, width=1 if flag else 0)),
                              customdata=np.c_[s_["customer_id"], s_["alert_score"]],
                              hovertemplate="%{customdata[0]}<br>IF percentile %{x:.1f}<br>Inflow/income %{y:.1f}x"
                                            "<br>Alert score %{customdata[1]:.0f}<extra></extra>")
        fig.update_yaxes(type="log")
        st.plotly_chart(style(fig, 340, xtitle="Isolation Forest percentile", ytitle="Inflow ÷ income (log)"),
                        width="stretch")

    st.markdown("### Alert queue")
    st.dataframe(alerts, hide_index=True, width="stretch", height=280,
                 column_config={"alert_score": st.column_config.ProgressColumn("Alert score", min_value=0,
                                                                               max_value=100, format="%.0f")})
    if len(alerts):
        pick = st.selectbox("Inspect suspicious transactions for", alerts["customer_id"].tolist())
        tx = q("SELECT txn_date, txn_hour, amount, direction, channel, category, counterparty_id, "
               "amount_zscore, alert_reason FROM txn_alerts WHERE customer_id = :c ORDER BY txn_date", c=pick)
        if tx.empty:
            st.info("Flagged on behaviour pattern (model score); no single transaction breaches a rule.")
        else:
            st.dataframe(tx, hide_index=True, width="stretch")

# ============================================================================= 6. STRESS TESTING
with tabs[5]:
    coefs = q("SELECT * FROM macro_coefficients").iloc[0].to_dict()
    stress = q("SELECT * FROM stress_results")
    scen_order = list(SCENARIOS)
    tot = stress.groupby("scenario")[["ead", "ecl_stressed", "ecl_base"]].sum().reindex(scen_order)

    k = st.columns(4)
    base_ecl = tot.loc["Baseline", "ecl_stressed"]
    k[0].metric("Baseline ECL", cr(base_ecl))
    k[1].metric("Adverse ECL", cr(tot.loc["Adverse", "ecl_stressed"]),
                f"+{tot.loc['Adverse', 'ecl_stressed'] / base_ecl - 1:.0%}", delta_color="inverse")
    k[2].metric("Severe ECL", cr(tot.loc["Severe", "ecl_stressed"]),
                f"+{tot.loc['Severe', 'ecl_stressed'] / base_ecl - 1:.0%}", delta_color="inverse")
    k[3].metric("Satellite model R²", f"{coefs['r_squared']:.2f}", "logit(default rate) ~ macro", delta_color="off", delta_arrow="off")

    a, b = st.columns([1.3, 1])
    with a:
        st.markdown("### Stressed ECL by product")
        fig = go.Figure()
        for sname in scen_order:
            s_ = stress[stress["scenario"] == sname].set_index("product").reindex(PRODUCTS)
            fig.add_bar(x=PRODUCTS, y=s_["ecl_stressed"] / 1e7, name=sname, marker_color=SCEN_COLOR[sname],
                        customdata=np.c_[s_["pd_base"], s_["pd_stressed"], s_["lgd_stressed"], s_["stage2_ead_share"]],
                        hovertemplate="%{x} · " + sname + "<br>ECL ₹%{y:,.2f} cr<br>PD %{customdata[0]:.2%} → "
                                      "%{customdata[1]:.2%}<br>LGD %{customdata[2]:.1%}<br>Stage 2 share "
                                      "%{customdata[3]:.1%}<extra></extra>")
        fig.update_layout(barmode="group", bargroupgap=0.06)
        st.plotly_chart(style(fig, 360, ytitle="ECL (₹ crore)"), width="stretch")
    with b:
        st.markdown("### Scenarios")
        sc_tab = pd.DataFrame(SCENARIOS).T[MACRO_VARS + ["property", "vehicle", "gold"]]
        sc_tab.columns = ["Δ GDP (pp)", "Δ Unemp (pp)", "Δ Repo (pp)", "Δ HPI (pp)", "Property", "Vehicle", "Gold"]
        st.dataframe(sc_tab, width="stretch")
        for n_, s_ in SCENARIOS.items():
            st.markdown(f'<div class="muted"><b>{n_}:</b> {s_["narrative"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="muted" style="margin-top:8px">Satellite β: GDP {coefs["gdp_growth"]:+.3f} · '
                    f'Unemp {coefs["unemployment_rate"]:+.3f} · Repo {coefs["repo_rate"]:+.3f} · '
                    f'HPI {coefs["hpi_growth"]:+.3f} (per pp, on log-odds)</div>', unsafe_allow_html=True)

    st.markdown("### Build a custom scenario")

    @st.cache_data(ttl=600)
    def loan_base():
        return q("""SELECT l.*, e.loan_dpd, e.ecl AS ecl_current, r.pd_12m, r.ews_level, r.current_dpd
                    FROM loans l JOIN ecl_results e ON e.loan_id = l.loan_id
                    JOIN customer_risk r ON r.customer_id = l.customer_id""")

    s1, s2, s3, s4, s5, s6 = st.columns(6)
    custom = {
        "gdp_growth": s1.slider("Δ GDP growth (pp)", -10.0, 3.0, -2.0, 0.5),
        "unemployment_rate": s2.slider("Δ Unemployment (pp)", -2.0, 8.0, 1.5, 0.5),
        "repo_rate": s3.slider("Δ Repo rate (pp)", -2.0, 3.0, 0.5, 0.25),
        "hpi_growth": s4.slider("Δ House-price growth (pp)", -25.0, 5.0, -5.0, 1.0),
        "property": s5.slider("Property value shock", -0.40, 0.10, -0.05, 0.05),
        "vehicle": s6.slider("Vehicle value shock", -0.40, 0.10, -0.05, 0.05),
        "gold": 0.0,
    }
    lb = loan_base()
    res = apply_scenario(lb.drop(columns=["ecl_current"]), coefs, custom)
    res["ecl_current"] = lb["ecl_current"].to_numpy()
    by = res.groupby("product").agg(ecl_base=("ecl_current", "sum"), ecl_custom=("ecl", "sum")).reindex(PRODUCTS)
    shift = sum(coefs[v] * custom[v] for v in MACRO_VARS)
    a, b = st.columns([1, 1.4])
    with a:
        tb, tc = by["ecl_base"].sum(), by["ecl_custom"].sum()
        st.metric("Custom-scenario ECL", cr(tc), f"{tc / tb - 1:+.1%} vs baseline", delta_color="inverse")
        st.metric("Log-odds shift (before product sensitivity)", f"{shift:+.3f}",
                  f"≈ {np.exp(shift):.2f}× odds of default", delta_color="off", delta_arrow="off")
        st.metric("Stage 2 + 3 share of EAD",
                  f"{res.loc[res['stage'] >= 2, 'ead'].sum() / res['ead'].sum():.1%}")
    with b:
        fig = go.Figure()
        fig.add_bar(x=PRODUCTS, y=by["ecl_base"] / 1e7, name="Baseline", marker_color=SCEN_COLOR["Baseline"])
        fig.add_bar(x=PRODUCTS, y=by["ecl_custom"] / 1e7, name="Custom", marker_color=SCEN_COLOR["Custom"])
        fig.update_layout(barmode="group", bargroupgap=0.06)
        st.plotly_chart(style(fig, 300, ytitle="ECL (₹ crore)"), width="stretch")

    st.markdown("### Macro history used to fit the satellite model")
    mh = q("SELECT * FROM macro_history ORDER BY quarter")
    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Scatter(x=mh["quarter"], y=mh["portfolio_default_rate"], mode="lines",
                                   line=dict(color=SERIES[0], width=2),
                                   hovertemplate="%{x}<br>Default rate %{y:.2%}<extra></extra>"))
        fig.update_layout(title=dict(text="Portfolio default rate", font=dict(size=13, color=INK)))
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(style(fig, 260, legend=False), width="stretch")
    with b:
        labels = {"gdp_growth": "Real GDP growth (% YoY)", "unemployment_rate": "Unemployment rate (%)",
                  "repo_rate": "RBI repo rate (%)", "hpi_growth": "House-price growth (% YoY)"}
        var = st.selectbox("Macro driver", MACRO_VARS, format_func=labels.get)
        fig = go.Figure(go.Scatter(x=mh["quarter"], y=mh[var], mode="lines", line=dict(color=SERIES[1], width=2),
                                   hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>"))
        fig.update_layout(title=dict(text=labels[var], font=dict(size=13, color=INK)))
        st.plotly_chart(style(fig, 260, legend=False), width="stretch")
    st.caption("Stylised quarterly series approximating India 2015–2025 (incl. the 2020 COVID shock); "
               "the portfolio default-rate series is synthetic.")

# ============================================================================= 7. MODEL PERFORMANCE
with tabs[6]:
    met = q("SELECT * FROM model_metrics")
    test = met[met["dataset"] == "test"].set_index("model")
    pop = report.get("population", {})
    calib = report.get("calibration", {})
    st.markdown(f'<div class="muted">Performing borrowers only ({pop.get("n_performing", 0):,}); target = 90+ DPD within '
                f'12 months (default rate {pop.get("default_rate", 0):.2%}). Stratified 60/20/20 train / validation / '
                f'test split; F1 threshold tuned on validation; all metrics below on the untouched test set. '
                f'Platt recalibration applied to champion: <b>{"yes" if calib.get("platt_applied") else "no"}</b> '
                f'(cross-fitted ECE {calib.get("cv_ece_raw", 0):.4f} raw vs {calib.get("cv_ece_platt", 0):.4f} Platt).'
                f'</div>', unsafe_allow_html=True)
    cols = ["roc_auc", "gini", "pr_auc", "ks", "f1", "precision", "recall", "brier", "log_loss", "ece", "hl_pvalue"]
    st.dataframe(test[cols].rename(columns={"roc_auc": "ROC-AUC", "gini": "Gini", "pr_auc": "PR-AUC", "ks": "KS",
                                            "f1": "F1", "brier": "Brier", "log_loss": "Log-loss", "ece": "ECE",
                                            "hl_pvalue": "HL p-value"}).style.format("{:.3f}")
                 .highlight_max(subset=["ROC-AUC", "Gini", "PR-AUC", "KS", "F1"], color="#cde2fb")
                 .highlight_min(subset=["Brier", "Log-loss", "ECE"], color="#cde2fb"), width="stretch")

    curves = report.get("curves", {})
    models = [m for m in ["Logistic Regression", "Random Forest", "XGBoost", "XGBoost (Platt)"] if m in curves]
    a, b, c3 = st.columns(3)
    with a:
        fig = go.Figure()
        for i, m in enumerate(models):
            fig.add_scatter(x=curves[m]["roc"]["fpr"], y=curves[m]["roc"]["tpr"], mode="lines",
                            name=f"{m} ({test.loc[m, 'roc_auc']:.3f})",
                            line=dict(color=SERIES[i], width=2, dash="dot" if "Platt" in m else "solid"))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=BASELINE, dash="dash", width=1),
                        showlegend=False, hoverinfo="skip")
        st.markdown("### ROC curve")
        style(fig, 360, xtitle="False positive rate", ytitle="True positive rate")
        fig.update_layout(legend=dict(y=0.03, x=0.98, xanchor="right", yanchor="bottom", orientation="v",
                                      bgcolor="rgba(252,252,251,0.9)"))
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = go.Figure()
        for i, m in enumerate(models):
            fig.add_scatter(x=curves[m]["pr"]["recall"], y=curves[m]["pr"]["precision"], mode="lines",
                            name=f"{m} ({test.loc[m, 'pr_auc']:.3f})",
                            line=dict(color=SERIES[i], width=2, dash="dot" if "Platt" in m else "solid"))
        st.markdown("### Precision–recall")
        style(fig, 360, xtitle="Recall", ytitle="Precision")
        fig.update_layout(legend=dict(y=0.98, x=0.98, xanchor="right", yanchor="top", orientation="v",
                                      bgcolor="rgba(252,252,251,0.9)"))
        st.plotly_chart(fig, width="stretch")
    with c3:
        champ_name = "XGBoost (Platt)" if calib.get("platt_applied") else "XGBoost"
        fig = go.Figure()
        for i, m in enumerate(["Logistic Regression", champ_name]):
            cal = curves[m]["calibration"]
            fig.add_scatter(x=cal["mean_pd"], y=cal["observed"], mode="lines+markers", name=m,
                            line=dict(color=SERIES[0 if i == 0 else 2], width=2), marker=dict(size=8),
                            hovertemplate="Predicted %{x:.2%}<br>Observed %{y:.2%}<extra>" + m + "</extra>")
        mx = max(max(curves[champ_name]["calibration"]["observed"]), max(curves[champ_name]["calibration"]["mean_pd"]))
        fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=BASELINE, dash="dash", width=1),
                        showlegend=False, hoverinfo="skip")
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        st.markdown("### Calibration (deciles)")
        style(fig, 360, xtitle="Mean predicted PD", ytitle="Observed default rate")
        fig.update_layout(legend=dict(y=0.98, x=0.02, xanchor="left", yanchor="top", orientation="v",
                                      bgcolor="rgba(252,252,251,0.9)"))
        st.plotly_chart(fig, width="stretch")

    a, b = st.columns([1, 1])
    with a:
        st.markdown("### Gains, lift and KS by decile (champion)")
        dec = pd.DataFrame(curves[champ_name]["deciles"])
        fig = go.Figure()
        fig.add_bar(x=dec["decile"], y=dec["default_rate"], name="Default rate", marker_color=SERIES[0],
                    customdata=np.c_[dec["lift"], dec["cum_capture"], dec["mean_pd"]],
                    hovertemplate="Decile %{x}<br>Default rate %{y:.2%}<br>Mean PD %{customdata[2]:.2%}"
                                  "<br>Lift %{customdata[0]:.1f}x<br>Cum. capture %{customdata[1]:.0%}<extra></extra>")
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(dtick=1, title="Score decile (1 = riskiest)")
        st.plotly_chart(style(fig, 320, legend=False, ytitle="Observed default rate"), width="stretch")
        d_show = dec[["decile", "n", "defaults", "mean_pd", "default_rate", "lift", "cum_capture", "ks"]]
        st.dataframe(d_show.style.format({"mean_pd": "{:.2%}", "default_rate": "{:.2%}", "lift": "{:.2f}",
                                          "cum_capture": "{:.1%}", "ks": "{:.3f}"}),
                     hide_index=True, width="stretch", height=250)
    with b:
        st.markdown("### Global feature importance (mean |SHAP|)")
        imp = q("SELECT * FROM shap_importance ORDER BY rank").head(15).iloc[::-1]
        fig = go.Figure(go.Bar(y=imp["feature"], x=imp["mean_abs_shap"], orientation="h", marker_color=SERIES[0],
                               hovertemplate="%{y}<br>mean |SHAP| %{x:.3f}<extra></extra>"))
        st.plotly_chart(style(fig, 560, legend=False, xtitle="Mean |SHAP| (log-odds)"), width="stretch")

st.divider()
st.caption("Synthetic data generated for demonstration. PD: XGBoost champion vs Logistic Regression and Random Forest "
           "benchmarks · ECL = PD × LGD × EAD with IFRS 9 / Ind AS 109 staging · EWS aligned to RBI SMA framework.")
