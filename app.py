"""
智图忆市 v2.0 - K线形态匹配与量化分析平台
© 2026 智图忆市 保留所有权利
功能：形态匹配 / 形态检测 / 概率统计 / 策略回测 / 多标的对比
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import csv
import io
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.spatial.distance import cosine
from datetime import datetime, timedelta

# ===================== 页面设置 =====================
st.set_page_config(
    page_title="智图忆市 Pro - K线形态量化分析",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== 样式 =====================
st.markdown("""
<style>
    .main-header {font-size:2rem; font-weight:700; color:#1f77b4; text-align:center;}
    .sub-header {font-size:1.1rem; color:#666; text-align:center; margin-bottom:20px;}
    .metric-card {background:#f0f7ff; padding:15px; border-radius:10px; border-left:4px solid #1f77b4;}
    .match-card {background:#fffef0; padding:12px; border-radius:8px; margin:8px 0; border-left:4px solid #ffa500;}
    .up {color:#e24a4a; font-weight:bold;}
    .down {color:#2e7d32; font-weight:bold;}
    .neutral {color:#888;}
    .stDeployButton {display:none !important;}
    footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ===================== 股票列表 =====================
STOCK_MAP = {
    # 主要指数
    "【指数】上证指数": "sh000001", "【指数】深证成指": "sz399001",
    "【指数】创业板指": "sz399006", "【指数】科创50": "sh000688",
    "【指数】沪深300": "sh000300", "【指数】中证500": "sh000905",
    "【指数】上证50": "sh000016", "【指数】中证1000": "sh000852",
    # 蓝筹
    "【蓝筹】贵州茅台": "sh600519", "【蓝筹】宁德时代": "sz300750",
    "【蓝筹】比亚迪": "sz002594", "【蓝筹】东方财富": "sz300059",
    "【蓝筹】中国平安": "sh601318", "【蓝筹】招商银行": "sh600036",
    "【蓝筹】五粮液": "sz000858", "【蓝筹】美的集团": "sz000333",
    "【蓝筹】工商银行": "sh601398", "【蓝筹】农业银行": "sh601288",
    "【蓝筹】中国石油": "sh601857", "【蓝筹】中国石化": "sh600028",
    # 科技
    "【科技】海康威视": "sz002415", "【科技】中芯国际": "sh688981",
    "【科技】寒武纪": "sh688256", "【科技】科大讯飞": "sz002230",
    # 金融科技
    "【金融科技】同花顺": "sz300033", "【金融科技】恒生电子": "sh600570",
    "【金融科技】顶点软件": "sh603383",
}

# ===================== 数据源 =====================
@st.cache_data(ttl=3600)
def get_sina_data(code, datalen=800):
    """新浪行情接口"""
    try:
        url = f"https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?symbol={code}&scale=240&datalen={datalen}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if not data:
            return None, "新浪接口返回空数据"
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["day"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600)
def get_akshare_data(code, datalen=800):
    """AKShare接口作为备选"""
    try:
        import akshare as ak
        symbol = code
        if code.startswith("sh"):
            symbol = f"sh{code[2:]}"
        elif code.startswith("sz"):
            symbol = f"sz{code[2:]}"
        
        df = ak.stock_zh_a_hist(symbol=symbol[2:], period="daily", 
                                  start_date=(datetime.now() - timedelta(days=datalen)).strftime("%Y%m%d"),
                                  end_date=datetime.now().strftime("%Y%m%d"), adjust="hfq")
        df["date"] = pd.to_datetime(df["日期"])
        df = df.rename(columns={"开盘": "open", "最高": "high", "最低": "low", 
                                 "收盘": "close", "成交量": "volume"})
        df = df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
        return df, None
    except Exception as e:
        return None, str(e)

def get_stock_data(code, datalen=800):
    """双保险数据获取"""
    df, err = get_sina_data(code, datalen)
    if df is not None and len(df) > 50:
        return df, "sina"
    
    df2, err2 = get_akshare_data(code, datalen)
    if df2 is not None and len(df2) > 50:
        return df2, "akshare"
    
    return None, f"新浪:{err} | AKShare:{err2}"

# ===================== 技术指标 =====================
def add_indicators(df):
    """计算技术指标"""
    df = df.copy()
    close = df["close"]
    
    # 均线
    df["ma5"] = close.rolling(5).mean()
    df["ma10"] = close.rolling(10).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi"] = 100 - (100 / (rs + 1))
    
    # 成交量均线
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    
    return df

# ===================== 形态检测 =====================
def detect_patterns(df, window=20):
    """检测常见技术形态"""
    if len(df) < window:
        return {}
    
    recent = df.tail(window).copy()
    close = recent["close"].values
    high = recent["high"].values
    low = recent["low"].values
    volume = recent["volume"].values
    
    patterns = {}
    
    # 1. 双底形态
    try:
        min_idx = np.argmin(low)
        if min_idx > 3 and min_idx < window - 5:
            left_low = low[:min_idx].min()
            right_low = low[min_idx:].min()
            left_high_after = high[:min_idx].max()
            if abs(left_low - right_low) / (left_low + 1) < 0.02:  # 两底接近
                patterns["双底"] = {
                    "置信度": 0.85,
                    "描述": f"在 {window} 天窗口内形成双底结构，两低点差 < 2%",
                    "信号": "看涨"
                }
    except:
        pass
    
    # 2. 双顶形态
    try:
        max_idx = np.argmax(high)
        if max_idx > 3 and max_idx < window - 5:
            left_high_val = high[:max_idx].max()
            right_high_val = high[max_idx:].max()
            if abs(left_high_val - right_high_val) / (left_high_val + 1) < 0.02:
                patterns["双顶"] = {
                    "置信度": 0.85,
                    "描述": f"在 {window} 天窗口内形成双顶结构，两高点差 < 2%",
                    "信号": "看跌"
                }
    except:
        pass
    
    # 3. 三角形整理
    try:
        slope_high = (high[-1] - high[0]) / window
        slope_low = (low[-1] - low[0]) / window
        if abs(slope_high) < 0.01 and abs(slope_low) > 0.01 and slope_low > 0:
            patterns["上升三角形"] = {
                "置信度": 0.75,
                "描述": "低点逐步抬高，高点横向整理，通常向上突破",
                "信号": "看涨"
            }
        elif abs(slope_low) < 0.01 and abs(slope_high) > 0.01 and slope_high < 0:
            patterns["下降三角形"] = {
                "置信度": 0.75,
                "描述": "高点逐步降低，低点横向整理，通常向下突破",
                "信号": "看跌"
            }
    except:
        pass
    
    # 4. 均线多头排列
    ma5 = recent["ma5"].values[-1] if "ma5" in recent.columns else 0
    ma10 = recent["ma10"].values[-1] if "ma10" in recent.columns else 0
    ma20 = recent["ma20"].values[-1] if "ma20" in recent.columns else 0
    
    if ma5 > ma10 > ma20 > 0:
        patterns["均线多头"] = {
            "置信度": 0.80,
            "描述": "MA5 > MA10 > MA20，多头趋势强劲",
            "信号": "看涨"
        }
    elif ma5 < ma10 < ma20:
        patterns["均线空头"] = {
            "置信度": 0.80,
            "描述": "MA5 < MA10 < MA20，空头趋势明显",
            "信号": "看跌"
        }
    
    # 5. RSI 超买超卖
    rsi = recent["rsi"].values[-1] if "rsi" in recent.columns else 50
    if rsi < 30:
        patterns["RSI超卖"] = {
            "置信度": 0.70,
            "描述": f"RSI = {rsi:.1f}，低于30，超卖区域",
            "信号": "可能反弹"
        }
    elif rsi > 70:
        patterns["RSI超买"] = {
            "置信度": 0.70,
            "描述": f"RSI = {rsi:.1f}，高于70，超买区域",
            "信号": "可能回调"
        }
    
    # 6. MACD 金叉死叉
    if "macd" in recent.columns and "macd_signal" in recent.columns:
        macd = recent["macd"].values
        sig = recent["macd_signal"].values
        if len(macd) >= 2:
            if macd[-2] < sig[-2] and macd[-1] > sig[-1]:
                patterns["MACD金叉"] = {
                    "置信度": 0.75,
                    "描述": "MACD 从下方穿越信号线，看涨信号",
                    "信号": "看涨"
                }
            elif macd[-2] > sig[-2] and macd[-1] < sig[-1]:
                patterns["MACD死叉"] = {
                    "置信度": 0.75,
                    "描述": "MACD 从上方穿越信号线，看跌信号",
                    "信号": "看跌"
                }
    
    return patterns

# ===================== 形态匹配核心 =====================
def normalize(series):
    s = series.values.astype(float)
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-8)

def find_similar_patterns(df, window, top_n=10):
    """滑动窗口匹配历史相似形态"""
    prices = df["close"].values
    n = len(prices)
    template = normalize(pd.Series(prices[-window:]))
    
    results = []
    for i in range(n - 2 * window):
        seg = normalize(pd.Series(prices[i:i+window]))
        sim = 1 - cosine(template, seg)
        
        # 计算后续走势
        future_prices = prices[i+window : i+window+30] if i+window+30 <= n else prices[i+window:]
        if len(future_prices) >= 5:
            future_return = (future_prices[-1] - prices[i+window-1]) / (prices[i+window-1] + 1e-8) * 100
            future_volatility = np.std(future_prices) / (np.mean(future_prices) + 1e-8) * 100
            
            # 计算持有不同天数的收益率
            returns_5d = (future_prices[min(4, len(future_prices)-1)] - prices[i+window-1]) / (prices[i+window-1] + 1e-8) * 100 if len(future_prices) >= 5 else 0
            returns_10d = (future_prices[min(9, len(future_prices)-1)] - prices[i+window-1]) / (prices[i+window-1] + 1e-8) * 100 if len(future_prices) >= 10 else 0
            returns_20d = (future_prices[min(19, len(future_prices)-1)] - prices[i+window-1]) / (prices[i+window-1] + 1e-8) * 100 if len(future_prices) >= 20 else 0
            
            results.append({
                "idx": i,
                "similarity": sim,
                "date": df.iloc[i]["date"],
                "start_price": prices[i],
                "end_price": prices[i+window-1],
                "future_end_price": future_prices[-1],
                "return_5d": returns_5d,
                "return_10d": returns_10d,
                "return_20d": returns_20d,
                "total_return": future_return,
                "volatility": future_volatility,
                "is_up": future_return > 0
            })
    
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_n]

# ===================== K线绘图 =====================
def plot_kline_with_pattern(df_seg, window, match_date=None, sim_score=None, 
                            show_ma=True, show_vol=True, show_macd=True, show_rsi=False,
                            title=""):
    """绘制专业K线图"""
    df_plot = df_seg.copy()
    
    # 涨跌幅颜色
    colors = ["#e24a4a" if df_plot.iloc[i]["close"] >= df_plot.iloc[i]["open"] else "#2e7d32" 
              for i in range(len(df_plot))]
    
    fig = make_subplots(
        rows=3 if show_macd else 2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2] if show_macd else [0.75, 0.25],
        subplot_titles=("", "成交量", "MACD") if show_macd else ("", "成交量")
    )
    
    # K线
    fig.add_trace(go.Candlestick(
        x=df_plot["date"],
        open=df_plot["open"],
        high=df_plot["high"],
        low=df_plot["low"],
        close=df_plot["close"],
        increasing_line_color="#e24a4a",
        decreasing_line_color="#2e7d32",
        name="K线"
    ), row=1, col=1)
    
    # 均线
    if show_ma and "ma5" in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["ma5"], 
                                  line=dict(color="#ff9800", width=1), name="MA5"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["ma10"], 
                                  line=dict(color="#2196f3", width=1), name="MA10"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["ma20"], 
                                  line=dict(color="#9c27b0", width=1), name="MA20"), row=1, col=1)
    
    # 匹配区域标记
    if window > 0 and len(df_plot) > window:
        fig.add_vrect(
            x0=df_plot.iloc[window]["date"], x1=df_plot.iloc[-1]["date"],
            fillcolor="rgba(255, 165, 0, 0.1)",
            line_width=0,
            row="all", col=1
        )
        fig.add_vline(
            x=df_plot.iloc[window]["date"],
            line_dash="dash", line_color="#ffa500", line_width=2,
            row="all", col=1
        )
    
    # 成交量
    if show_vol:
        colors_vol = ["#e24a4a" if df_plot.iloc[i]["close"] >= df_plot.iloc[i]["open"] else "#2e7d32"
                      for i in range(len(df_plot))]
        fig.add_trace(go.Bar(x=df_plot["date"], y=df_plot["volume"],
                              marker_color=colors_vol, name="成交量",
                              hovertemplate="成交量: %{y:,.0f}<extra></extra>"), row=2, col=1)
        if "vol_ma5" in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["vol_ma5"],
                                     line=dict(color="#ff9800", width=1), name="Vol MA5"), row=2, col=1)
    
    # MACD
    if show_macd and "macd" in df_plot.columns:
        colors_macd = ["#e24a4a" if v >= 0 else "#2e7d32" for v in df_plot["macd_hist"]]
        fig.add_trace(go.Bar(x=df_plot["date"], y=df_plot["macd_hist"],
                             marker_color=colors_macd, name="MACD柱"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["macd"],
                                 line=dict(color="#1f77b4", width=1), name="DIF"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["macd_signal"],
                                 line=dict(color="#ff5722", width=1), name="DEA"), row=3, col=1)
    
    # 标题
    title_text = title
    if match_date:
        title_text += f" | 匹配日: {match_date.strftime('%Y-%m-%d')}"
    if sim_score:
        title_text += f" | 相似度: {sim_score:.2%}"
    
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        template="plotly_white",
        height=500 if show_macd else 400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        hovermode="x unified"
    )
    
    return fig

# ===================== 统计面板 =====================
def show_stats_panel(matches, window, extend_days):
    """显示统计面板"""
    if not matches:
        return
    
    up_count = sum(1 for m in matches if m["is_up"])
    total = len(matches)
    win_rate = up_count / total * 100
    
    avg_return_5d = np.mean([m["return_5d"] for m in matches])
    avg_return_10d = np.mean([m["return_10d"] for m in matches])
    avg_return_20d = np.mean([m["return_20d"] for m in matches])
    avg_total = np.mean([m["total_return"] for m in matches])
    avg_vol = np.mean([m["volatility"] for m in matches])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("历史匹配次数", f"{total} 次", f"窗口: {window}天")
    with col2:
        color = "up" if avg_return_5d > 0 else "down"
        st.metric("5日平均收益", f"{avg_return_5d:+.2f}%", 
                  "↑" if avg_return_5d > 0 else "↓")
    with col3:
        st.metric("10日平均收益", f"{avg_return_10d:+.2f}%",
                  "↑" if avg_return_10d > 0 else "↓")
    with col4:
        st.metric("20日平均收益", f"{avg_return_20d:+.2f}%",
                  "↑" if avg_return_20d > 0 else "↓")
    with col5:
        st.metric("上涨胜率", f"{win_rate:.0f}%", 
                  f"盈利 {up_count} 次 / {total} 次")
    
    # 收益分布图
    st.subheader("📊 历史匹配后续走势分布")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # 各持有期收益分布
        returns_data = {
            "持有期": ["5日", "10日", "20日", f"{extend_days}日"],
            "平均收益": [avg_return_5d, avg_return_10d, avg_return_20d, avg_total]
        }
        fig_bar = px.bar(
            x=returns_data["持有期"], 
            y=returns_data["平均收益"],
            color=returns_data["平均收益"],
            color_continuous_scale=["#2e7d32", "#e24a4a"],
            title="不同持有期平均收益(%)"
        )
        fig_bar.update_layout(showlegend=False, height=300)
        fig_bar.update_yaxes(title="收益(%)")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_right:
        # 收益分布直方图
        fig_hist = px.histogram(
            x=[m["total_return"] for m in matches],
            nbins=20,
            title="20日收益分布直方图",
            labels={"x": "收益(%)", "y": "次数"}
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_hist.add_vline(x=avg_total, line_dash="solid", line_color="#1f77b4",
                          annotation_text=f"均值: {avg_total:.1f}%")
        fig_hist.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # 详细匹配表
    st.subheader(f"📋 Top {len(matches)} 相似形态详情")
    
    detail_df = pd.DataFrame([{
        "序号": i+1,
        "匹配日期": m["date"].strftime("%Y-%m-%d"),
        "相似度": f"{m['similarity']:.2%}",
        "5日收益": f"{m['return_5d']:+.2f}%",
        "10日收益": f"{m['return_10d']:+.2f}%",
        "20日收益": f"{m['total_return']:+.2f}%",
        "波动率": f"{m['volatility']:.2f}%",
        "结果": "✅ 上涨" if m["is_up"] else "❌ 下跌"
    } for i, m in enumerate(matches)])
    
    st.dataframe(detail_df, use_container_width=True, hide_index=True)
    
    # 导出CSV
    csv_data = detail_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 导出结果为CSV",
        csv_data,
        f"形态匹配结果_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        key="download-csv"
    )

# ===================== 主界面 =====================
def main():
    # 标题
    st.markdown('<p class="main-header">🎯 智图忆市 Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">K线形态量化分析 · 历史相似匹配 · 概率统计决策</p>', unsafe_allow_html=True)
    
    # 免责声明
    st.warning("⚠️  本工具仅供学习复盘研究，不构成任何投资建议。股市有风险，投资需谨慎。")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 分析设置")
        
        # 标的
        st.subheader("📊 选择标的")
        selected = st.selectbox("股票/指数", list(STOCK_MAP.keys()), label_visibility="collapsed")
        code = STOCK_MAP[selected]
        
        st.subheader("🔧 形态匹配设置")
        pattern_days = st.slider("模板窗口天数", 5, 60, 20, help="选取最近多少根K线作为匹配模板")
        extend_days = st.slider("后续观察天数", 5, 60, 20, help="匹配后观察多少天的走势")
        top_n = st.slider("匹配数量", 3, 20, 10, help="显示最相似的多少个历史形态")
        max_history = st.slider("历史数据量", 300, 1000, 800, help="从过去多少天开始搜索")
        
        st.subheader("📐 图表设置")
        show_ma = st.checkbox("显示均线 (MA5/10/20)", True)
        show_vol = st.checkbox("显示成交量", True)
        show_macd = st.checkbox("显示MACD", True)
        
        st.subheader("🔍 形态检测")
        detect_pattern = st.checkbox("自动检测常见形态", True)
        
        find_btn = st.button("🔍 开始分析", type="primary", use_container_width=True)
    
    # 主内容
    if find_btn or "last_code" in st.session_state:
        if find_btn:
            st.session_state["last_code"] = code
            st.session_state["last_pattern"] = pattern_days
            st.session_state["last_extend"] = extend_days
            st.session_state["last_top_n"] = top_n
            st.session_state["last_max_history"] = max_history
        
        code = st.session_state.get("last_code", code)
        pattern_days = st.session_state.get("last_pattern", pattern_days)
        extend_days = st.session_state.get("last_extend", extend_days)
        top_n = st.session_state.get("last_top_n", top_n)
        max_history = st.session_state.get("last_max_history", max_history)
        
        with st.spinner("正在获取数据..."):
            df, source = get_stock_data(code, max_history)
        
        if df is None or len(df) < 100:
            st.error(f"❌ 数据获取失败，请稍后重试。代码: {code}")
            return
        
        st.success(f"✅ 数据加载成功 [{source.upper()}] 共 {len(df)} 根日K线 | {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")
        
        # 添加指标
        df = add_indicators(df)
        
        # ===== 第一部分：当前行情 =====
        st.markdown("---")
        st.subheader(f"📌 当前行情: {selected}")
        
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        
        price = last["close"]
        change = price - prev["close"]
        change_pct = change / prev["close"] * 100
        
        with col_info1:
            st.metric("最新价", f"{price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
        with col_info2:
            st.metric("最高价", f"{last['high']:.2f}")
        with col_info3:
            st.metric("最低价", f"{last['low']:.2f}")
        with col_info4:
            vol_ratio = last["volume"] / df["vol_ma5"].iloc[-1] if df["vol_ma5"].iloc[-1] > 0 else 1
            st.metric("量比", f"{vol_ratio:.2f}x", "较5日均量")
        
        # 当前K线图
        fig_current = plot_kline_with_pattern(
            df.tail(80), 0,
            show_ma=show_ma, show_vol=show_vol, show_macd=show_macd,
            title=f"{selected} - 近80日K线"
        )
        st.plotly_chart(fig_current, use_container_width=True)
        
        # ===== 第二部分：形态检测 =====
        if detect_pattern:
            st.markdown("---")
            st.subheader("🔍 当前形态检测")
            
            patterns = detect_patterns(df, window=pattern_days)
            
            if patterns:
                cols = st.columns(min(len(patterns), 4))
                for i, (name, info) in enumerate(patterns.items()):
                    with cols[i % 4]:
                        signal_color = "up" if "看涨" in info["信号"] else "down"
                        st.markdown(f"""
                        <div class="match-card">
                            <h4>{name}</h4>
                            <p class="{signal_color}">信号: {info['信号']}</p>
                            <p>置信度: {info['置信度']:.0%}</p>
                            <p style="font-size:0.85rem; color:#666;">{info['描述']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("未检测到明确形态")
        
        # ===== 第三部分：形态匹配 =====
        st.markdown("---")
        st.subheader(f"🔎 历史相似形态匹配 (模板: 近 {pattern_days} 日 → 观察: 后 {extend_days} 日)")
        
        with st.spinner("正在匹配历史形态..."):
            matches = find_similar_patterns(df, pattern_days, top_n)
        
        if matches:
            show_stats_panel(matches, pattern_days, extend_days)
            
            # 每个匹配的详细K线图
            st.markdown("---")
            st.subheader(f"📈 Top {len(matches)} 相似形态详细走势")
            
            for i, m in enumerate(matches):
                start_idx = m["idx"]
                end_idx = start_idx + pattern_days + extend_days
                seg_df = df.iloc[start_idx:end_idx].copy()
                
                return_color = "up" if m["is_up"] else "down"
                return_icon = "📈" if m["is_up"] else "📉"
                
                with st.expander(f"{return_icon} 匹配 #{i+1} | {m['date'].strftime('%Y-%m-%d')} | 相似度: {m['similarity']:.2%} | 20日收益: <span class='{return_color}'>{m['total_return']:+.2f}%</span>", expanded=(i < 3)):
                    fig_match = plot_kline_with_pattern(
                        seg_df, pattern_days,
                        match_date=m["date"], sim_score=m["similarity"],
                        show_ma=show_ma, show_vol=show_vol, show_macd=show_macd,
                        title=f"历史匹配 #{i+1}"
                    )
                    st.plotly_chart(fig_match, use_container_width=True)
                    
                    # 该匹配详细信息
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("5日收益", f"{m['return_5d']:+.2f}%")
                    with c2:
                        st.metric("10日收益", f"{m['return_10d']:+.2f}%")
                    with c3:
                        st.metric("20日收益", f"{m['total_return']:+.2f}%")
                    with c4:
                        st.metric("波动率", f"{m['volatility']:.2f}%")
        else:
            st.warning("未找到足够的匹配数据")
    
    else:
        # 初始界面
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("📊 **K线形态匹配**\n用最近N日走势，在历史上寻找相似形态")
        with col2:
            st.info("📐 **自动形态检测**\n智能识别双底、头肩顶、三角形等经典形态")
        with col3:
            st.info("📈 **概率统计分析**\n基于历史匹配计算上涨概率和预期收益")
        
        st.markdown("---")
        st.markdown("""
        ### 🚀 使用指南
        
        1. **选择标的** - 从左侧选择股票或指数
        2. **设置参数** - 调整模板天数和观察天数
        3. **开始分析** - 点击侧边栏按钮获取结果
        4. **查看详情** - 点击展开查看每个匹配的详细走势
        5. **导出数据** - 下载CSV进行进一步分析
        
        ### 💡 核心理念
        
        > "历史不会简单重复，但总押着韵脚"
        
        智图忆市通过量化相似形态的历史表现，为您的投资决策提供数据支持。
        """)
        
        # 演示数据
        if st.button("📊 查看演示效果"):
            st.info("请先点击侧边栏「开始分析」按钮")
    
    # 页脚
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#888; font-size:0.8rem;'>"
        "© 2026 智图忆市 | 数据来源: 新浪财经 / AKShare | 仅供学习研究"
        "</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
