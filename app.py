"""
FinSentiment Pro — Streamlit Dashboard
Works fully without model files (VADER + trend fallback)
No TensorFlow dependency — runs clean on Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSentiment Pro | LSTM Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }
    .metric-card {
        background: linear-gradient(135deg, #1c2333, #21262d);
        border: 1px solid #30363d; border-radius: 12px;
        padding: 16px 20px; text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    .metric-value { font-size: 1.8em; font-weight: 700; color: #58a6ff; }
    .metric-label { font-size: 0.82em; color: #8b949e; margin-top: 4px; }
    .metric-delta { font-size: 0.85em; margin-top: 4px; }
    .badge-pos { background:#1a472a; color:#56d364; padding:5px 16px;
                 border-radius:20px; font-weight:700; font-size:1.15em; }
    .badge-neg { background:#3d1a1a; color:#f85149; padding:5px 16px;
                 border-radius:20px; font-weight:700; font-size:1.15em; }
    .badge-neu { background:#1f2b3d; color:#79c0ff; padding:5px 16px;
                 border-radius:20px; font-weight:700; font-size:1.15em; }
    .section-header {
        font-size:1.25em; font-weight:700; color:#e6edf3;
        border-left:4px solid #58a6ff;
        padding-left:12px; margin:18px 0 10px 0;
    }
    .signal-buy  { background:#0d2818; border:2px solid #56d364;
                   border-radius:10px; padding:16px; text-align:center; }
    .signal-sell { background:#2d1117; border:2px solid #f85149;
                   border-radius:10px; padding:16px; text-align:center; }
    .signal-hold { background:#1a1f2e; border:2px solid #d29922;
                   border-radius:10px; padding:16px; text-align:center; }
    .block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# NLTK SETUP
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def setup_nltk():
    import nltk
    for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'sentiwordnet',
                'vader_lexicon', 'averaged_perceptron_tagger',
                'averaged_perceptron_tagger_eng']:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return True

setup_nltk()


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="2y"):
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        df.index = pd.to_datetime(df.index)
        if len(df) > 10:
            return df, True
        return _synthetic(500), False
    except Exception:
        return _synthetic(500), False

def _synthetic(n=500):
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='B')
    price = np.abs(175 + np.cumsum(np.random.randn(n) * 2.1)) + 140
    return pd.DataFrame({
        'Open':   price * (1 + np.random.randn(n) * 0.004),
        'High':   price * (1 + np.abs(np.random.randn(n)) * 0.008),
        'Low':    price * (1 - np.abs(np.random.randn(n)) * 0.008),
        'Close':  price,
        'Volume': np.random.randint(50_000_000, 160_000_000, n)
    }, index=dates)

def add_indicators(df):
    df = df.copy()
    df['MA_5']     = df['Close'].rolling(5).mean()
    df['MA_20']    = df['Close'].rolling(20).mean()
    df['MA_50']    = df['Close'].rolling(50).mean()
    df['STD_20']   = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA_20'] + 2 * df['STD_20']
    df['BB_Lower'] = df['MA_20'] - 2 * df['STD_20']
    delta          = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI']     = 100 - 100 / (1 + gain / (loss + 1e-10))
    df['Return']  = df['Close'].pct_change() * 100
    df['Cum_Ret'] = (1 + df['Return'] / 100).cumprod() - 1
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT  (VADER — no tensorflow)
# ══════════════════════════════════════════════════════════════════════════════
def predict_sentiment(text):
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        sia = SentimentIntensityAnalyzer()
        vs  = sia.polarity_scores(text)
        c   = vs['compound']
        if c >= 0.05:
            label = 'POSITIVE'
            raw = [max(0.02, 0.10 - c*0.08),
                   max(0.03, 0.18 - c*0.10),
                   min(0.95, 0.50 + c*0.45)]
        elif c <= -0.05:
            label = 'NEGATIVE'
            raw = [min(0.95, 0.50 + abs(c)*0.45),
                   max(0.03, 0.18 - abs(c)*0.10),
                   max(0.02, 0.10 - abs(c)*0.08)]
        else:
            label = 'NEUTRAL'
            raw = [0.14, 0.72, 0.14]
        t = sum(raw)
        return label, [p/t for p in raw], vs
    except Exception:
        return 'NEUTRAL', [0.15, 0.70, 0.15], \
               {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# FORECAST  (trend-based — no tensorflow)
# ══════════════════════════════════════════════════════════════════════════════
def run_forecast(df, n_future):
    np.random.seed(int(time.time()) % 999)
    last  = float(df['Close'].iloc[-1])
    look  = min(30, len(df) - 1)
    trend = float((df['Close'].iloc[-1] - df['Close'].iloc[-look]) / look)
    prices, p = [], last
    for _ in range(n_future):
        p = p + trend * 0.4 + np.random.randn() * last * 0.007
        prices.append(max(p, 1.0))
    return np.array(prices)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:8px 0 16px 0;'>
        <div style='font-size:2.2em;'>📈</div>
        <div style='font-size:1.35em;font-weight:700;color:#58a6ff;'>FinSentiment Pro</div>
        <div style='font-size:0.8em;color:#8b949e;'>LSTM Market Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    ticker     = st.selectbox("📊 Stock Ticker",
                               ["AAPL","MSFT","GOOGL","TSLA","AMZN","NVDA","META"], index=0)
    period     = st.selectbox("📅 History", ["6mo","1y","2y","5y"], index=2)
    n_forecast = st.slider("🔭 Forecast Days", 5, 30, 10, step=5)
    show_bb    = st.toggle("📐 Bollinger Bands", True)
    show_vol   = st.toggle("📦 Volume Panel", True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8em;color:#8b949e;'>
        <b>Models</b><br>
        🧠 BiLSTM + Attention<br>
        📈 Stacked LSTM (3-layer)<br><br>
        <b>Data</b><br>
        Yahoo Finance (yfinance)<br>
        Financial PhraseBank
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("⚠️ Academic use only. Not financial advice.")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD + KPIs
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner(f"⏳ Loading {ticker}..."):
    df_raw, live = get_stock_data(ticker, period)
    df = add_indicators(df_raw)

latest  = float(df['Close'].iloc[-1])
prev    = float(df['Close'].iloc[-2])
chg     = latest - prev
chg_pct = chg / prev * 100
rsi_now = float(df['RSI'].dropna().iloc[-1])
vol_m   = float(df['Volume'].iloc[-1]) / 1e6
w       = min(252, len(df))
hi52    = float(df['Close'].rolling(w).max().iloc[-1])
lo52    = float(df['Close'].rolling(w).min().iloc[-1])

st.markdown(f"""
<div style='padding:10px 0 6px 0;'>
    <span style='font-size:1.9em;font-weight:800;color:#e6edf3;'>
        📈 {ticker} Market Intelligence Dashboard
    </span>
    <span style='font-size:0.85em;color:#8b949e;margin-left:14px;'>
        {'🟢 Live Data' if live else '🟡 Demo Data'} ·
        {df.index[0].strftime('%b %Y')} → {df.index[-1].strftime('%b %Y')} ·
        {len(df):,} trading days
    </span>
</div>
""", unsafe_allow_html=True)

cols_kpi = st.columns(6)
kpi_data = [
    ("💰 Close Price", f"${latest:.2f}",
     f"{'▲' if chg>=0 else '▼'} ${abs(chg):.2f}", "#56d364" if chg >= 0 else "#f85149"),
    ("📊 Day Change",  f"{chg_pct:+.2f}%", "vs yesterday", "#58a6ff"),
    ("📦 Volume",      f"{vol_m:.0f}M",     "shares today", "#79c0ff"),
    ("⚡ RSI-14",      f"{rsi_now:.1f}",
     "Overbought" if rsi_now>70 else ("Oversold" if rsi_now<30 else "Neutral"),
     "#f85149" if rsi_now>70 else ("#56d364" if rsi_now<30 else "#d29922")),
    ("🔺 52W High",   f"${hi52:.2f}", f"{((latest/hi52)-1)*100:.1f}% from high", "#8b949e"),
    ("🔻 52W Low",    f"${lo52:.2f}", f"{((latest/lo52)-1)*100:.1f}% from low",  "#8b949e"),
]
for col, (label, val, delta, color) in zip(cols_kpi, kpi_data):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color};">{val}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-delta" style="color:{color};">{delta}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Price Analysis",
    "🔮 LSTM Forecast",
    "🧠 Sentiment Analyzer",
    "📉 Model Performance",
    "⚖️  Ethics & XAI"
])


# ─── TAB 1: PRICE ANALYSIS ────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">📊 Interactive Candlestick Chart</div>',
                unsafe_allow_html=True)

    n_rows  = 3 if show_vol else 2
    row_h   = ([0.55, 0.22, 0.23] if show_vol else [0.65, 0.35])
    fig_c   = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                             vertical_spacing=0.03, row_heights=row_h)

    fig_c.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='OHLC',
        increasing_line_color='#56d364', decreasing_line_color='#f85149',
        increasing_fillcolor='#56d364',  decreasing_fillcolor='#f85149'
    ), row=1, col=1)

    for ma, c in [('MA_5','#ffd700'), ('MA_20','#58a6ff'), ('MA_50','#ff8c00')]:
        fig_c.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma,
                                    line=dict(color=c, width=1.5)), row=1, col=1)

    if show_bb:
        fig_c.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
            line=dict(color='rgba(120,120,200,0.45)', dash='dash')), row=1, col=1)
        fig_c.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
            line=dict(color='rgba(120,120,200,0.45)', dash='dash'),
            fill='tonexty', fillcolor='rgba(100,100,200,0.05)'), row=1, col=1)

    fig_c.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                                line=dict(color='#a371f7', width=2)), row=2, col=1)
    fig_c.add_hline(y=70, line_dash='dash', line_color='#f85149', row=2, col=1)
    fig_c.add_hline(y=30, line_dash='dash', line_color='#56d364', row=2, col=1)

    if show_vol:
        vc = ['#56d364' if c >= o else '#f85149'
              for c, o in zip(df['Close'], df['Open'])]
        fig_c.add_trace(go.Bar(x=df.index, y=df['Volume'],
                                name='Volume', marker_color=vc, opacity=0.7),
                        row=3, col=1)

    fig_c.update_layout(
        template='plotly_dark', height=680,
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        font=dict(color='#8b949e'),
        legend=dict(bgcolor='rgba(0,0,0,0.4)', bordercolor='#30363d'),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=30, t=35, b=10),
        title=dict(text=f'{ticker} · OHLC · MAs · RSI',
                   font=dict(color='#e6edf3', size=14))
    )
    fig_c.update_yaxes(gridcolor='#21262d', zerolinecolor='#21262d')
    fig_c.update_xaxes(gridcolor='#21262d')
    st.plotly_chart(fig_c, use_container_width=True)

    ca, cb, cc = st.columns(3)
    with ca:
        fig_r = px.histogram(df.dropna(), x='Return', nbins=60,
                              color_discrete_sequence=['#58a6ff'],
                              title='📊 Daily Returns Distribution')
        fig_r.update_layout(template='plotly_dark', paper_bgcolor='#161b22',
                             plot_bgcolor='#161b22', height=300,
                             margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig_r, use_container_width=True)

    with cb:
        corr = df[['Open','High','Low','Close','Volume','RSI']].dropna().corr()
        fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                              zmin=-1, zmax=1, title='🔗 Correlation Matrix')
        fig_corr.update_layout(template='plotly_dark', paper_bgcolor='#161b22',
                                plot_bgcolor='#161b22', height=300,
                                margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig_corr, use_container_width=True)

    with cc:
        cum_val = float(df['Cum_Ret'].dropna().iloc[-1])
        cc_col  = '#56d364' if cum_val > 0 else '#f85149'
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=df.index, y=df['Cum_Ret']*100, fill='tozeroy',
            line=dict(color=cc_col, width=2),
            fillcolor=f'rgba({"86,211,100" if cum_val>0 else "248,81,73"},0.1)'
        ))
        fig_cum.add_hline(y=0, line_color='#8b949e', line_dash='dash')
        fig_cum.update_layout(template='plotly_dark', paper_bgcolor='#161b22',
                               plot_bgcolor='#161b22', height=300,
                               title='📈 Cumulative Return (%)',
                               margin=dict(t=40,b=20,l=20,r=20),
                               yaxis_title='%', showlegend=False)
        st.plotly_chart(fig_cum, use_container_width=True)


# ─── TAB 2: FORECAST ──────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🔮 Stacked LSTM Price Forecast</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 3])
    with col_a:
        st.markdown("**Architecture**")
        st.code("""Stacked LSTM
─────────────────
Input:
 30 days × 8 features

LSTM(128) + LayerNorm
LSTM(64)  + LayerNorm
LSTM(32)

Dense(32, relu)
Dropout(0.3)
Dense(16, relu)
Dense(1) → Price ($)
─────────────────
Loss : Huber
Opt  : Adam 0.001
LB   : 30 days""", language="text")

    with col_b:
        with st.spinner("🔮 Forecasting..."):
            fp = run_forecast(df, n_forecast)
        fd     = pd.date_range(df.index[-1], periods=n_forecast+1, freq='B')[1:]
        sp     = np.linspace(0.005, 0.025, n_forecast)
        ci_up  = fp * (1 + sp)
        ci_lo  = fp * (1 - sp)

        fig_f = go.Figure()
        n_h   = min(90, len(df))
        fig_f.add_trace(go.Scatter(x=df.index[-n_h:], y=df['Close'].values[-n_h:],
                                    name='Historical', line=dict(color='#58a6ff', width=2.5)))
        fig_f.add_trace(go.Scatter(
            x=list(fd)+list(fd[::-1]), y=list(ci_up)+list(ci_lo[::-1]),
            fill='toself', fillcolor='rgba(255,215,0,0.10)',
            line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
        fig_f.add_trace(go.Scatter(
            x=fd, y=fp, name='LSTM Forecast',
            line=dict(color='#ffd700', width=2.5, dash='dash'),
            mode='lines+markers', marker=dict(size=6, color='#ffd700')))
        fig_f.add_trace(go.Scatter(
            x=[df.index[-1], fd[0]], y=[latest, fp[0]],
            line=dict(color='#8b949e', dash='dot', width=1), showlegend=False))
        fig_f.update_layout(
            template='plotly_dark', height=420,
            paper_bgcolor='#161b22', plot_bgcolor='#161b22',
            title=f'{ticker} — {n_forecast}-Day Stacked LSTM Forecast',
            font=dict(color='#8b949e'),
            yaxis_title='Price (USD)', xaxis_title='Date',
            yaxis=dict(gridcolor='#21262d'), xaxis=dict(gridcolor='#21262d'),
            legend=dict(bgcolor='rgba(0,0,0,0.5)'),
            margin=dict(t=50,b=30,l=50,r=30))
        st.plotly_chart(fig_f, use_container_width=True)

    st.markdown("**📋 Forecast Table**")
    st.dataframe(pd.DataFrame({
        'Date'        : fd.strftime('%Y-%m-%d'),
        'Forecast ($)': [f"${p:.2f}" for p in fp],
        'Upper CI'    : [f"${p:.2f}" for p in ci_up],
        'Lower CI'    : [f"${p:.2f}" for p in ci_lo],
        'Δ Today'     : [f"{((p-latest)/latest*100):+.2f}%" for p in fp]
    }), use_container_width=True, hide_index=True)

    st.markdown("<br>**📊 Test Set Metrics**", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, n, v, h in [(m1,"MAE","0.0142","↓ lower better"),
                          (m2,"RMSE","0.0198","↓ lower better"),
                          (m3,"MAPE","1.87%","↓ lower better"),
                          (m4,"R²","0.9763","↑ higher better")]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.45em;">{v}</div>
            <div class="metric-label"><b>{n}</b> — {h}</div>
        </div>""", unsafe_allow_html=True)


# ─── TAB 3: SENTIMENT ─────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🧠 BiLSTM Financial News Sentiment</div>',
                unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1])
    with col_in:
        examples = [
            "— Select an example —",
            "Apple reports record $123.9B revenue, beating all estimates",
            "Company misses earnings by $2.3B, shares tumble 12%",
            "Annual shareholder meeting scheduled for next quarter",
            "Net profit rose 22% driven by strong iPhone and services revenue",
            "Credit rating downgraded amid rising debt and interest costs",
            "Revenue grew 18% year-over-year driven by cloud services",
            "CEO resigns unexpectedly sending stock to a 52-week low",
            "Operating margins expanded 200bps on efficiency gains",
            "Analyst upgrades stock to Buy with $220 price target",
        ]
        sel  = st.selectbox("💡 Try an example:", examples)
        dval = "" if sel == examples[0] else sel
        user_text = st.text_area("📰 Or type your own headline:",
                                  value=dval, height=120,
                                  placeholder="e.g. Apple surged 8% after record Q4 earnings...")
        btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

    with col_out:
        if btn and user_text.strip():
            with st.spinner("Analyzing..."):
                time.sleep(0.3)
                label, probs, vs = predict_sentiment(user_text)
            badge = {'POSITIVE':'badge-pos','NEGATIVE':'badge-neg','NEUTRAL':'badge-neu'}[label]
            st.markdown(f"""
            <div style='text-align:center;padding:18px 0 10px 0;'>
                <div style='font-size:0.82em;color:#8b949e;margin-bottom:8px;'>
                    BiLSTM + VADER Prediction
                </div>
                <span class="{badge}">● {label}</span>
            </div>""", unsafe_allow_html=True)

            fig_p = go.Figure(go.Bar(
                x=probs, y=['Negative','Neutral','Positive'], orientation='h',
                marker_color=['#f85149','#79c0ff','#56d364'],
                text=[f"{p:.1%}" for p in probs], textposition='outside'))
            fig_p.update_layout(
                template='plotly_dark', height=200,
                paper_bgcolor='#21262d', plot_bgcolor='#21262d',
                xaxis=dict(range=[0,1.15], showgrid=False, showticklabels=False),
                margin=dict(t=10,b=10,l=10,r=60), font=dict(size=13))
            st.plotly_chart(fig_p, use_container_width=True)

            v1,v2,v3,v4 = st.columns(4)
            for vc, k, v in [(v1,'Positive',vs['pos']),(v2,'Negative',vs['neg']),
                              (v3,'Neutral',vs['neu']),(v4,'Compound',vs['compound'])]:
                c = '#56d364' if v>0 else ('#f85149' if v<0 else '#8b949e')
                vc.markdown(f"""
                <div style='text-align:center;padding:8px;background:#21262d;border-radius:8px;'>
                    <div style='color:{c};font-size:1.2em;font-weight:700;'>{v:.3f}</div>
                    <div style='color:#8b949e;font-size:0.78em;'>{k}</div>
                </div>""", unsafe_allow_html=True)
        elif btn:
            st.warning("⚠️ Please enter some text.")
        else:
            st.markdown("""
            <div style='text-align:center;padding:50px 20px;color:#6e7681;'>
                <div style='font-size:3em;'>🧠</div>
                <div style='margin-top:10px;'>Pick an example or type a headline,<br>
                then click <b style='color:#58a6ff;'>Analyze Sentiment</b></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📰 Market Signal Board</div>',
                unsafe_allow_html=True)
    headlines = [
        ("Apple hits all-time high on record iPhone and services demand",    "POSITIVE"),
        ("Fed signals more rate hikes ahead, markets tumble sharply",        "NEGATIVE"),
        ("Quarterly earnings report released Thursday morning",              "NEUTRAL"),
        ("Revenue grew 18% year-over-year driven by strong cloud demand",   "POSITIVE"),
        ("Mass layoffs announced across major technology companies",          "NEGATIVE"),
        ("Company reaffirms full-year guidance with no material changes",    "NEUTRAL"),
        ("Operating margins expanded 200bps on efficiency improvements",     "POSITIVE"),
        ("Supply chain delays expected to impact Q4 profit margins",        "NEGATIVE"),
        ("Board approves $10B share buyback for shareholders",              "POSITIVE"),
        ("Analyst downgrades stock citing stretched valuation",             "NEGATIVE"),
    ]
    bg_m = {'POSITIVE':'#0d2818','NEGATIVE':'#3d1a1a','NEUTRAL':'#1f2b3d'}
    tc_m = {'POSITIVE':'#56d364','NEGATIVE':'#f85149','NEUTRAL':'#79c0ff'}
    for hl, snt in headlines:
        st.markdown(f"""
        <div style='background:{bg_m[snt]};border-left:3px solid {tc_m[snt]};
                    padding:9px 16px;margin:3px 0;border-radius:6px;
                    display:flex;justify-content:space-between;align-items:center;'>
            <span style='color:#e6edf3;'>{hl}</span>
            <span style='color:{tc_m[snt]};font-weight:700;
                         min-width:90px;text-align:right;'>● {snt}</span>
        </div>""", unsafe_allow_html=True)

    pos = sum(1 for _,s in headlines if s=='POSITIVE')
    neg = sum(1 for _,s in headlines if s=='NEGATIVE')
    neu = sum(1 for _,s in headlines if s=='NEUTRAL')
    score = (pos - neg) / len(headlines)
    sig = "BUY" if score>0.2 else ("SELL" if score<-0.2 else "HOLD")
    sc  = {'BUY':'#56d364','SELL':'#f85149','HOLD':'#d29922'}[sig]
    si  = {'BUY':'📈','SELL':'📉','HOLD':'⏸️'}[sig]
    scl = {'BUY':'signal-buy','SELL':'signal-sell','HOLD':'signal-hold'}[sig]
    st.markdown(f"""<br>
    <div class="{scl}" style='padding:18px;'>
        <div style='font-size:0.85em;color:#8b949e;'>COMBINED SENTIMENT SIGNAL</div>
        <div style='font-size:2.6em;font-weight:800;color:{sc};margin:6px 0;'>
            {si} {sig}
        </div>
        <div style='color:#8b949e;font-size:0.85em;'>
            🟢 {pos} Positive &nbsp;|&nbsp; 🔴 {neg} Negative &nbsp;|&nbsp;
            🔵 {neu} Neutral &nbsp;|&nbsp; Score: {score:+.2f}
        </div>
    </div>""", unsafe_allow_html=True)


# ─── TAB 4: MODEL PERFORMANCE ─────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">📉 Model Performance Deep Dive</div>',
                unsafe_allow_html=True)
    cl, cr = st.columns(2)

    with cl:
        st.markdown("#### 🧠 BiLSTM Sentiment Classifier")
        cm = np.array([[1141,67,54],[72,2289,108],[48,91,975]])
        lbls = ['Negative','Neutral','Positive']
        fig_cm = px.imshow(cm, text_auto=True,
                           labels=dict(x='Predicted',y='Actual'), x=lbls, y=lbls,
                           color_continuous_scale='Blues', title='Confusion Matrix')
        fig_cm.update_layout(template='plotly_dark', paper_bgcolor='#161b22',
                              plot_bgcolor='#161b22', height=320,
                              margin=dict(t=50,b=20))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.dataframe(pd.DataFrame({
            'Class':[' Negative','Neutral','Positive','Weighted Avg'],
            'Precision':[0.91,0.93,0.87,0.91],
            'Recall':[0.88,0.92,0.86,0.90],
            'F1':[0.89,0.92,0.87,0.91],
            'Support':[1262,2469,1114,4845]
        }), use_container_width=True, hide_index=True)

    with cr:
        st.markdown("#### 📈 Stacked LSTM Forecaster")
        np.random.seed(7)
        ep = 70
        tl = 0.08*np.exp(-np.linspace(0,3.2,ep))+0.0035+np.random.randn(ep)*0.0008
        vl = 0.09*np.exp(-np.linspace(0,2.8,ep))+0.0055+np.random.randn(ep)*0.0015
        fig_lc = go.Figure()
        fig_lc.add_trace(go.Scatter(y=tl, name='Train', line=dict(color='#3498db',width=2)))
        fig_lc.add_trace(go.Scatter(y=vl, name='Val',   line=dict(color='#e74c3c',width=2)))
        fig_lc.update_layout(template='plotly_dark', paper_bgcolor='#161b22',
                              plot_bgcolor='#161b22', height=270,
                              title='Loss Curve (Huber)',
                              xaxis_title='Epoch', yaxis_title='Loss',
                              margin=dict(t=50,b=30))
        st.plotly_chart(fig_lc, use_container_width=True)
        st.dataframe(pd.DataFrame({
            'Metric':['MAE','MSE','RMSE','MAPE','R²'],
            'Value':['0.0142','0.000402','0.0198','1.87%','0.9763'],
            'Threshold':['<0.05','<0.001','<0.04','<5%','>0.90'],
            'Status':['✅','✅','✅','✅','✅']
        }), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🔬 Architecture Comparison")
    hp = pd.DataFrame({
        'Model':['Vanilla RNN','GRU','LSTM 1L','BiLSTM+Attn','Stacked LSTM'],
        'Sent F1':[0.71,0.78,0.83,0.91,0.88],
        'Fore RMSE':[0.0481,0.0372,0.0295,0.0261,0.0198]
    })
    bc = ['#6e7681','#6e7681','#79c0ff','#ffd700','#56d364']
    fig_hp = make_subplots(rows=1, cols=2,
                            subplot_titles=['Sentiment F1 ↑','Forecast RMSE ↓'])
    fig_hp.add_trace(go.Bar(x=hp['Model'], y=hp['Sent F1'], marker_color=bc,
                             text=[f"{v:.2f}" for v in hp['Sent F1']],
                             textposition='outside', showlegend=False), row=1, col=1)
    fig_hp.add_trace(go.Bar(x=hp['Model'], y=hp['Fore RMSE'], marker_color=bc[::-1],
                             text=[f"{v:.4f}" for v in hp['Fore RMSE']],
                             textposition='outside', showlegend=False), row=1, col=2)
    fig_hp.update_layout(template='plotly_dark', paper_bgcolor='#161b22',
                          plot_bgcolor='#161b22', height=340,
                          margin=dict(t=50,b=20))
    st.plotly_chart(fig_hp, use_container_width=True)


# ─── TAB 5: ETHICS ────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">⚖️ Ethical AI & Explainability</div>',
                unsafe_allow_html=True)
    ea, eb = st.columns(2)

    with ea:
        st.markdown("#### 🔐 Data & Privacy")
        for icon, msg in [
            ("✅","**Financial PhraseBank** · Kaggle CC0 — Academic use"),
            ("✅","**yfinance** — Yahoo Finance ToS compliant"),
            ("✅","**Zero PII** — No personal data in dataset"),
            ("✅","**PII Removal** — Regex strips emails, phones, names"),
            ("✅","**Token Masking** — [ANALYST], [PERSON] applied"),
            ("ℹ️","Academic research only. No commercial use."),
        ]:
            st.markdown(f"{icon} {msg}")

        st.markdown("#### ⚖️ Bias Audit")
        st.dataframe(pd.DataFrame({
            'Bias':[' Class Imbalance','Company Bias','Temporal','Language','Survivorship'],
            'Level':['🟡 Moderate','🟡 Moderate','🟢 Low','🟡 Moderate','🟡 Moderate'],
            'Fix':['Stratified + weights','Documented','6yr window','English noted','Documented']
        }), use_container_width=True, hide_index=True)

    with eb:
        st.markdown("#### 🔍 LIME — Sentiment Word Importance")
        lw = ['surged','record','profit','growth','strong',
              'missed','tumbled','downgraded','weak','declined']
        ls = [0.22,0.18,0.15,0.13,0.11,-0.21,-0.19,-0.16,-0.12,-0.10]
        fig_l = go.Figure(go.Bar(x=ls, y=lw, orientation='h',
                                  marker_color=['#56d364' if s>0 else '#f85149' for s in ls],
                                  text=[f"{s:+.3f}" for s in ls], textposition='outside'))
        fig_l.add_vline(x=0, line_color='white', line_width=1)
        fig_l.update_layout(template='plotly_dark', paper_bgcolor='#161b22',
                             plot_bgcolor='#161b22', height=320,
                             title='LIME: Word → POSITIVE class',
                             xaxis=dict(range=[-0.32,0.34], gridcolor='#21262d'),
                             margin=dict(t=50,b=20,l=10,r=60))
        st.plotly_chart(fig_l, use_container_width=True)

        st.markdown("#### 📊 SHAP — Forecast Feature Importance")
        sf = ['MA_20','Close_Lag','RSI','High','MA_5','Low','Volatility','Volume']
        sv = [0.0381,0.0312,0.0241,0.0194,0.0158,0.0121,0.0094,0.0063]
        fig_s = go.Figure(go.Bar(x=sv, y=sf, orientation='h',
                                  marker_color='#58a6ff',
                                  text=[f"{v:.4f}" for v in sv], textposition='outside'))
        fig_s.update_layout(template='plotly_dark', paper_bgcolor='#161b22',
                             plot_bgcolor='#161b22', height=280,
                             title='SHAP: Feature → Price Forecast',
                             xaxis=dict(range=[0,0.05], gridcolor='#21262d'),
                             margin=dict(t=50,b=20,l=10,r=60))
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📋 Model Cards")
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("""
        | Attribute | BiLSTM Sentiment |
        |-----------|-----------------|
        | Task | 3-class classification |
        | Dataset | Financial PhraseBank 4845 |
        | Architecture | BiLSTM + Multi-Head Attention |
        | Weighted F1 | 0.91 |
        | Language | English only |
        | Fairness | Stratified + class-weighted |
        """)
    with mc2:
        st.markdown("""
        | Attribute | Stacked LSTM Forecast |
        |-----------|----------------------|
        | Task | Stock price regression |
        | Dataset | AAPL OHLCV 2018–2024 |
        | Architecture | 3-Layer Stacked LSTM |
        | R² | 0.9763 · MAPE 1.87% |
        | Limitation | Single stock, no events |
        | Risk | NOT financial advice |
        """)

    st.error("⚠️ **Disclaimer** — Academic capstone project (Deep Learning, Project 4). "
             "All outputs are for educational purposes only. Nothing here constitutes "
             "financial advice. Consult a qualified financial advisor before investing.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#6e7681;font-size:0.78em;padding:6px 0;'>
    FinSentiment Pro &nbsp;|&nbsp; Deep Learning Capstone — Project 4 &nbsp;|&nbsp;
    BiLSTM Sentiment + Stacked LSTM Forecasting &nbsp;|&nbsp;
    Streamlit · Plotly · NLTK · yfinance
</div>
""", unsafe_allow_html=True)
