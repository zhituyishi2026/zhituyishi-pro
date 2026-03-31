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
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import io as io_module

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
    # 国内指数
    "上证指数": "sh000001",
    "深证成指": "sz399001",
    "创业板指": "sz399006",
    "科创50": "sh000688",
    "沪深300": "sh000300",
    "中证500": "sh000905",
    "上证50": "sh000016",
    "中证1000": "sh000852",
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
    """检测25种常见技术形态"""
    if len(df) < window:
        return {}
    
    recent = df.tail(window).copy()
    close = recent["close"].values
    high = recent["high"].values
    low = recent["low"].values
    volume = recent["volume"].values
    open_price = recent["open"].values
    
    patterns = {}
    
    # 1. 双底形态 (Double Bottom)
    try:
        min_idx = np.argmin(low)
        if min_idx > 3 and min_idx < window - 5:
            left_low = low[:min_idx].min()
            right_low = low[min_idx:].min()
            if abs(left_low - right_low) / (left_low + 1) < 0.02:
                patterns["双底"] = {
                    "置信度": 0.85,
                    "描述": f"在 {window} 天窗口内形成双底结构，两低点差 < 2%",
                    "信号": "看涨"
                }
    except:
        pass
    
    # 2. 双顶形态 (Double Top)
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
    
    # 3. 上升三角形 (Rising Triangle)
    # 4. 下降三角形 (Falling Triangle)
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
    
    # 5. 头肩顶 (Head and Shoulders Top)
    # 6. 头肩底 (Head and Shoulders Bottom)
    try:
        if window >= 10:
            mid_idx = window // 2
            left_shoulder = high[:mid_idx].max()
            head = high[mid_idx:].max()
            right_shoulder = high[-mid_idx:].max()
            
            if head > left_shoulder and head > right_shoulder and abs(left_shoulder - right_shoulder) / (left_shoulder + 1) < 0.03:
                patterns["头肩顶"] = {
                    "置信度": 0.78,
                    "描述": "形成左肩-头-右肩结构，中间高点最高，看跌信号",
                    "信号": "看跌"
                }
            
            # 头肩底
            left_low = low[:mid_idx].min()
            head_low = low[mid_idx:].min()
            right_low = low[-mid_idx:].min()
            
            if head_low < left_low and head_low < right_low and abs(left_low - right_low) / (left_low + 1) < 0.03:
                patterns["头肩底"] = {
                    "置信度": 0.78,
                    "描述": "形成左肩-头-右肩结构，中间低点最低，看涨信号",
                    "信号": "看涨"
                }
    except:
        pass
    
    # 7. 旗形整理 (Bull Flag / Bear Flag)
    try:
        if window >= 8:
            mid = window // 2
            first_half_range = high[:mid].max() - low[:mid].min()
            second_half_range = high[mid:].max() - low[mid:].min()
            
            if second_half_range < first_half_range * 0.5:
                if close[-1] > close[0]:
                    patterns["上升旗形"] = {
                        "置信度": 0.72,
                        "描述": "前期上升后进入整理，波幅收窄，看涨延续信号",
                        "信号": "看涨"
                    }
                else:
                    patterns["下降旗形"] = {
                        "置信度": 0.72,
                        "描述": "前期下跌后进入整理，波幅收窄，看跌延续信号",
                        "信号": "看跌"
                    }
    except:
        pass
    
    # 8. 楔形整理 (Rising Wedge / Falling Wedge)
    try:
        if window >= 8:
            slope_high = (high[-1] - high[0]) / window
            slope_low = (low[-1] - low[0]) / window
            
            if slope_high > 0 and slope_low > 0 and slope_high > slope_low:
                patterns["上升楔形"] = {
                    "置信度": 0.68,
                    "描述": "高低点都上升但高点上升更快，形成收窄楔形，看跌信号",
                    "信号": "看跌"
                }
            elif slope_high < 0 and slope_low < 0 and slope_high < slope_low:
                patterns["下降楔形"] = {
                    "置信度": 0.68,
                    "描述": "高低点都下降但低点下降更快，形成收窄楔形，看涨信号",
                    "信号": "看涨"
                }
    except:
        pass
    
    # 9. 矩形整理 (Rectangle)
    try:
        high_range = high.max() - high.min()
        low_range = low.max() - low.min()
        
        if high_range < (high.mean() * 0.03) and low_range < (low.mean() * 0.03):
            patterns["矩形整理"] = {
                "置信度": 0.70,
                "描述": "价格在一定范围内横向整理，突破方向决定后续走势",
                "信号": "中性"
            }
    except:
        pass
    
    # 10. 杯柄形态 (Cup and Handle)
    try:
        if window >= 12:
            cup_start = 0
            cup_bottom = low[:window//2].min()
            cup_end = window // 2
            handle_start = cup_end
            handle_bottom = low[handle_start:].min()
            
            if cup_bottom < handle_bottom and handle_bottom > cup_bottom * 0.98:
                patterns["杯柄形态"] = {
                    "置信度": 0.75,
                    "描述": "形成杯形底部后，右侧形成较浅的把手，看涨信号",
                    "信号": "看涨"
                }
    except:
        pass
    
    # 11. 吞没形态 (Engulfing)
    try:
        if len(close) >= 2:
            # 看涨吞没
            if open_price[-2] > close[-2] and close[-1] > open_price[-1] and \
               open_price[-1] <= close[-2] and close[-1] >= open_price[-2]:
                patterns["看涨吞没"] = {
                    "置信度": 0.76,
                    "描述": "前日阴线被今日阳线完全吞没，看涨反转信号",
                    "信号": "看涨"
                }
            # 看跌吞没
            elif open_price[-2] < close[-2] and close[-1] < open_price[-1] and \
                 open_price[-1] >= close[-2] and close[-1] <= open_price[-2]:
                patterns["看跌吞没"] = {
                    "置信度": 0.76,
                    "描述": "前日阳线被今日阴线完全吞没，看跌反转信号",
                    "信号": "看跌"
                }
    except:
        pass
    
    # 12. 早晨之星 (Morning Star)
    # 13. 黄昏之星 (Evening Star)
    try:
        if len(close) >= 3:
            # 早晨之星：阴线 -> 小星线 -> 阳线
            if open_price[-3] > close[-3] and \
               abs(close[-2] - open_price[-2]) < (close[-3] - open_price[-3]) * 0.3 and \
               close[-1] > open_price[-1]:
                patterns["早晨之星"] = {
                    "置信度": 0.74,
                    "描述": "阴线-小星线-阳线组合，底部反转看涨信号",
                    "信号": "看涨"
                }
            # 黄昏之星：阳线 -> 小星线 -> 阴线
            elif open_price[-3] < close[-3] and \
                 abs(close[-2] - open_price[-2]) < (close[-3] - open_price[-3]) * 0.3 and \
                 close[-1] < open_price[-1]:
                patterns["黄昏之星"] = {
                    "置信度": 0.74,
                    "描述": "阳线-小星线-阴线组合，顶部反转看跌信号",
                    "信号": "看跌"
                }
    except:
        pass
    
    # 14. 锤子线 (Hammer)
    # 15. 上吊线 (Hanging Man)
    try:
        if len(close) >= 1:
            body = abs(close[-1] - open_price[-1])
            lower_shadow = open_price[-1] - low[-1] if open_price[-1] > low[-1] else close[-1] - low[-1]
            upper_shadow = high[-1] - close[-1] if close[-1] > open_price[-1] else high[-1] - open_price[-1]
            
            if lower_shadow > body * 2 and upper_shadow < body * 0.5:
                if close[-1] > open_price[-1]:
                    patterns["锤子线"] = {
                        "置信度": 0.72,
                        "描述": "下影线长，实体小，上影线短，底部反转看涨信号",
                        "信号": "看涨"
                    }
                else:
                    patterns["上吊线"] = {
                        "置信度": 0.72,
                        "描述": "下影线长，实体小，上影线短，顶部反转看跌信号",
                        "信号": "看跌"
                    }
    except:
        pass
    
    # 16. 射击之星 (Shooting Star)
    try:
        if len(close) >= 1:
            body = abs(close[-1] - open_price[-1])
            upper_shadow = high[-1] - max(close[-1], open_price[-1])
            lower_shadow = min(close[-1], open_price[-1]) - low[-1]
            
            if upper_shadow > body * 2 and lower_shadow < body * 0.5 and close[-1] < open_price[-1]:
                patterns["射击之星"] = {
                    "置信度": 0.71,
                    "描述": "上影线长，实体小，下影线短，顶部反转看跌信号",
                    "信号": "看跌"
                }
    except:
        pass
    
    # 17. 孕育线 (Harami)
    try:
        if len(close) >= 2:
            prev_body = abs(close[-2] - open_price[-2])
            curr_body = abs(close[-1] - open_price[-1])
            
            if curr_body < prev_body * 0.5:
                # 看涨孕育线
                if open_price[-2] > close[-2] and close[-1] > open_price[-1]:
                    patterns["看涨孕育线"] = {
                        "置信度": 0.68,
                        "描述": "前日阴线内包含今日阳线，底部反转看涨信号",
                        "信号": "看涨"
                    }
                # 看跌孕育线
                elif open_price[-2] < close[-2] and close[-1] < open_price[-1]:
                    patterns["看跌孕育线"] = {
                        "置信度": 0.68,
                        "描述": "前日阳线内包含今日阴线，顶部反转看跌信号",
                        "信号": "看跌"
                    }
    except:
        pass
    
    # 18. 三白兵 (Three White Soldiers)
    # 19. 三乌鸦 (Three Black Crows)
    try:
        if len(close) >= 3:
            # 三白兵：连续三根阳线，收盘价逐日上升
            if all(close[i] > open_price[i] for i in range(-3, 0)) and \
               close[-3] < close[-2] < close[-1]:
                patterns["三白兵"] = {
                    "置信度": 0.77,
                    "描述": "连续三根阳线，收盘价逐日上升，强势看涨信号",
                    "信号": "看涨"
                }
            # 三乌鸦：连续三根阴线，收盘价逐日下降
            elif all(close[i] < open_price[i] for i in range(-3, 0)) and \
                 close[-3] > close[-2] > close[-1]:
                patterns["三乌鸦"] = {
                    "置信度": 0.77,
                    "描述": "连续三根阴线，收盘价逐日下降，强势看跌信号",
                    "信号": "看跌"
                }
    except:
        pass
    
    # 20. 跳空缺口 (Gap Up / Gap Down)
    try:
        if len(close) >= 2:
            gap = close[-1] - close[-2]
            gap_pct = gap / close[-2] * 100
            
            if gap_pct > 1.5:
                patterns["向上跳空"] = {
                    "置信度": 0.65,
                    "描述": f"向上跳空缺口 {gap_pct:.2f}%，看涨信号",
                    "信号": "看涨"
                }
            elif gap_pct < -1.5:
                patterns["向下跳空"] = {
                    "置信度": 0.65,
                    "描述": f"向下跳空缺口 {gap_pct:.2f}%，看跌信号",
                    "信号": "看跌"
                }
    except:
        pass
    
    # 21. 均线多头排列
    # 22. 均线空头排列
    ma5 = recent["ma5"].values[-1] if "ma5" in recent.columns else 0
    ma10 = recent["ma10"].values[-1] if "ma10" in recent.columns else 0
    ma20 = recent["ma20"].values[-1] if "ma20" in recent.columns else 0
    ma60 = recent["ma60"].values[-1] if "ma60" in recent.columns else 0
    
    if ma5 > ma10 > ma20 > ma60 > 0:
        patterns["均线多头排列"] = {
            "置信度": 0.80,
            "描述": "MA5 > MA10 > MA20 > MA60，多头趋势强劲",
            "信号": "看涨"
        }
    elif ma5 < ma10 < ma20 < ma60:
        patterns["均线空头排列"] = {
            "置信度": 0.80,
            "描述": "MA5 < MA10 < MA20 < MA60，空头趋势明显",
            "信号": "看跌"
        }
    
    # 23. 均线金叉/银叉 (Golden Cross / Death Cross)
    try:
        if "ma5" in recent.columns and "ma10" in recent.columns and len(recent) >= 2:
            ma5_vals = recent["ma5"].values
            ma10_vals = recent["ma10"].values
            
            if ma5_vals[-2] < ma10_vals[-2] and ma5_vals[-1] > ma10_vals[-1]:
                patterns["均线金叉"] = {
                    "置信度": 0.75,
                    "描述": "MA5 从下方穿越 MA10，看涨信号",
                    "信号": "看涨"
                }
            elif ma5_vals[-2] > ma10_vals[-2] and ma5_vals[-1] < ma10_vals[-1]:
                patterns["均线死叉"] = {
                    "置信度": 0.75,
                    "描述": "MA5 从上方穿越 MA10，看跌信号",
                    "信号": "看跌"
                }
    except:
        pass
    
    # 24. RSI 超买超卖
    rsi = recent["rsi"].values[-1] if "rsi" in recent.columns else 50
    if rsi < 30:
        patterns["RSI超卖"] = {
            "置信度": 0.70,
            "描述": f"RSI = {rsi:.1f}，低于30，超卖区域，可能反弹",
            "信号": "看涨"
        }
    elif rsi > 70:
        patterns["RSI超买"] = {
            "置信度": 0.70,
            "描述": f"RSI = {rsi:.1f}，高于70，超买区域，可能回调",
            "信号": "看跌"
        }
    
    # 25. MACD 金叉死叉
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
    
    # 26. BOLL 突破 (Bollinger Bands Breakout)
    try:
        if "close" in recent.columns and len(recent) >= 20:
            close_vals = recent["close"].values
            ma20_val = np.mean(close_vals[-20:])
            std_val = np.std(close_vals[-20:])
            upper_band = ma20_val + 2 * std_val
            lower_band = ma20_val - 2 * std_val
            
            if close_vals[-1] > upper_band:
                patterns["BOLL上轨突破"] = {
                    "置信度": 0.68,
                    "描述": "价格突破布林带上轨，可能继续上升或回调",
                    "信号": "看涨"
                }
            elif close_vals[-1] < lower_band:
                patterns["BOLL下轨突破"] = {
                    "置信度": 0.68,
                    "描述": "价格突破布林带下轨，可能继续下跌或反弹",
                    "信号": "看跌"
                }
    except:
        pass
    
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

# ===================== 策略回测模块 =====================
def backtest_pattern_strategy(df, pattern_name, holding_days=20):
    """
    回测指定形态的策略表现
    
    参数：
    - df: 包含技术指标的数据框
    - pattern_name: 形态名称
    - holding_days: 持有天数
    
    返回：回测统计结果字典
    """
    try:
        if len(df) < holding_days + 20:
            return None
        
        # 检测所有历史形态出现位置
        trades = []
        
        for i in range(20, len(df) - holding_days):
            window_df = df.iloc[i-20:i].copy()
            patterns = detect_patterns(window_df, window=20)
            
            if pattern_name in patterns:
                # 形态出现，记录买入点
                buy_price = df.iloc[i]["close"]
                buy_date = df.iloc[i]["date"]
                
                # 计算持有期收益
                sell_idx = min(i + holding_days, len(df) - 1)
                sell_price = df.iloc[sell_idx]["close"]
                sell_date = df.iloc[sell_idx]["date"]
                
                profit = sell_price - buy_price
                profit_pct = profit / buy_price * 100
                
                trades.append({
                    "buy_date": buy_date,
                    "buy_price": buy_price,
                    "sell_date": sell_date,
                    "sell_price": sell_price,
                    "profit": profit,
                    "profit_pct": profit_pct,
                    "is_profitable": profit > 0,
                    "holding_days": (sell_date - buy_date).days
                })
        
        if not trades:
            return None
        
        # 计算统计指标
        total_trades = len(trades)
        profitable_trades = sum(1 for t in trades if t["is_profitable"])
        win_rate = profitable_trades / total_trades * 100
        
        profits = [t["profit_pct"] for t in trades]
        avg_profit = np.mean(profits)
        max_profit = np.max(profits)
        min_profit = np.min(profits)
        
        # 计算夏普比率（简化版）
        if len(profits) > 1:
            sharpe_ratio = np.mean(profits) / (np.std(profits) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 计算权益曲线
        equity = [100]  # 初始资金100
        for t in trades:
            equity.append(equity[-1] * (1 + t["profit_pct"] / 100))
        
        return {
            "pattern_name": pattern_name,
            "holding_days": holding_days,
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "max_profit": max_profit,
            "min_profit": min_profit,
            "sharpe_ratio": sharpe_ratio,
            "trades": trades,
            "equity_curve": equity
        }
    except Exception as e:
        st.error(f"回测出错: {str(e)}")
        return None

def show_backtest_results(backtest_result):
    """
    显示回测结果
    
    参数：
    - backtest_result: backtest_pattern_strategy 返回的结果字典
    """
    if backtest_result is None:
        st.warning("该形态历史数据不足，无法回测")
        return
    
    # 策略概要卡片
    st.subheader("📊 回测结果概要")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 交易次数", f"{backtest_result['total_trades']} 次")
    with col2:
        st.metric("📈 胜率", f"{backtest_result['win_rate']:.1f}%", 
                  f"{backtest_result['profitable_trades']}/{backtest_result['total_trades']}")
    with col3:
        st.metric("💰 平均收益", f"{backtest_result['avg_profit']:+.2f}%")
    with col4:
        st.metric("🚀 最大收益", f"{backtest_result['max_profit']:+.2f}%")
    with col5:
        st.metric("📊 夏普比率", f"{backtest_result['sharpe_ratio']:.2f}")
    
    # 权益曲线
    st.subheader("📈 策略权益曲线")
    equity_df = pd.DataFrame({
        "交易序号": range(len(backtest_result["equity_curve"])),
        "账户权益": backtest_result["equity_curve"]
    })
    
    fig_equity = px.line(
        equity_df,
        x="交易序号",
        y="账户权益",
        title=f"策略权益曲线 (初始资金: 100)",
        markers=True
    )
    fig_equity.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # 收益分布直方图
    st.subheader("📊 交易收益分布")
    profits = [t["profit_pct"] for t in backtest_result["trades"]]
    
    fig_hist = px.histogram(
        x=profits,
        nbins=15,
        title="交易收益分布直方图",
        labels={"x": "收益(%)", "y": "交易次数"}
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_hist.add_vline(x=backtest_result["avg_profit"], line_dash="solid", 
                       line_color="#1f77b4", annotation_text=f"均值: {backtest_result['avg_profit']:.1f}%")
    fig_hist.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # 交易明细表
    st.subheader("📋 回测交易明细")
    trades_df = pd.DataFrame([{
        "序号": i + 1,
        "买入日期": t["buy_date"].strftime("%Y-%m-%d"),
        "买入价": f"{t['buy_price']:.2f}",
        "卖出日期": t["sell_date"].strftime("%Y-%m-%d"),
        "卖出价": f"{t['sell_price']:.2f}",
        "收益%": f"{t['profit_pct']:+.2f}%",
        "结果": "✅ 盈利" if t["is_profitable"] else "❌ 亏损"
    } for i, t in enumerate(backtest_result["trades"])])
    
    st.dataframe(trades_df, use_container_width=True, hide_index=True)
    
    # 导出CSV按钮 - 使用session_state存储数据
    csv_data = trades_df.to_csv(index=False).encode("utf-8-sig")
    st.session_state["backtest_csv_bytes"] = csv_data
    st.session_state["backtest_csv_name"] = f"回测结果_{backtest_result['pattern_name']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

# ===================== PDF报告生成 =====================
def show_stats_panel(matches, window, extend_days):
    """显示统计面板"""
    if not matches:
        st.warning("未找到匹配数据")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("历史匹配次数", f"{len(matches)} 次", f"窗口: {window}天")
    
    with col2:
        returns_5d = [m.get("return_5d", 0) for m in matches]
        avg_5d = np.mean(returns_5d) if returns_5d else 0
        st.metric("5日平均收益", f"{avg_5d:+.2f}%", "↑" if avg_5d > 0 else "↓")
    
    with col3:
        returns_10d = [m.get("return_10d", 0) for m in matches]
        avg_10d = np.mean(returns_10d) if returns_10d else 0
        st.metric("10日平均收益", f"{avg_10d:+.2f}%", "↑" if avg_10d > 0 else "↓")
    
    with col4:
        returns_20d = [m.get("total_return", 0) for m in matches]
        avg_20d = np.mean(returns_20d) if returns_20d else 0
        up_count = sum(1 for r in returns_20d if r > 0)
        win_rate = up_count / len(returns_20d) * 100 if returns_20d else 0
        st.metric("上涨胜率", f"{win_rate:.0f}%", f"盈利 {up_count} 次 / {len(matches)} 次")

def plot_kline_with_pattern(df, window, match_date=None, sim_score=None, show_ma=True, show_vol=True, show_macd=True, title="K线图", dark_mode=True):
    """绘制K线图表，默认深色模式"""
    # 固定深色配色，与页面背景统一
    bg_color = "#0e1117"
    grid_color = "#30363d"
    text_color = "#c9d1d9"
    up_color = "#3fb950"      # 上涨 - 绿色
    down_color = "#f85149"    # 下跌 - 红色
    ma_colors = ["#ff9800", "#58a6ff", "#bc8cff"]  # MA5/10/20
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # K线配色：根据涨跌设置颜色
    colors = []
    for i in range(len(df)):
        if df["close"].iloc[i] >= df["open"].iloc[i]:
            colors.append(up_color)
        else:
            colors.append(down_color)
    
    # K线
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            increasing_line_color=up_color,
            decreasing_line_color=down_color,
            increasing_fillcolor=up_color,
            decreasing_fillcolor=down_color,
        ),
        row=1, col=1
    )
    
    # 均线
    if show_ma:
        for idx, (ma_col, ma_name) in enumerate([("ma5", "MA5"), ("ma10", "MA10"), ("ma20", "MA20")]):
            if ma_col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["date"], y=df[ma_col], name=ma_name, mode="lines",
                        line=dict(color=ma_colors[idx], width=1.5)
                    ),
                    row=1, col=1
                )
    
    # 成交量
    if show_vol:
        vol_colors = [up_color if df["close"].iloc[i] >= df["open"].iloc[i] else down_color
                      for i in range(len(df))]
        fig.add_trace(
            go.Bar(x=df["date"], y=df["volume"], name="成交量", marker_color=vol_colors),
            row=2, col=1
        )
    
    # MACD
    if show_macd and "macd" in df.columns:
        macd_vals = df["macd"].values
        sig_vals = df["macd_signal"].values
        macd_colors = [up_color if macd_vals[i] >= sig_vals[i] else down_color
                       for i in range(len(macd_vals))]
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["macd"], name="MACD", mode="lines",
                       line=dict(color="#2196f3", width=1.5)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["macd_signal"], name="Signal", mode="lines",
                       line=dict(color="#ff9800", width=1.5)),
            row=3, col=1
        )
        # MACD 柱状图
        macd_hist_colors = [up_color if v >= 0 else down_color for v in df["macd_hist"].values]
        fig.add_trace(
            go.Bar(x=df["date"], y=df["macd_hist"], name="MACD Hist",
                   marker_color=macd_hist_colors, opacity=0.6),
            row=3, col=1
        )
    
    # 深色模式下自定义 K线 颜色映射（只更新 Candlestick trace）
    fig.update_traces(
        increasing_line_color=up_color,
        decreasing_line_color=down_color,
        increasing_fillcolor=up_color,
        decreasing_fillcolor=down_color,
        selector={"type": "candlestick"}
    )
    
    fig.update_layout(
        height=500,
        title_text=title,
        hovermode="x unified",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, color=text_color),
        yaxis=dict(gridcolor=grid_color, color=text_color),
        yaxis2=dict(gridcolor=grid_color, color=text_color),
        yaxis3=dict(gridcolor=grid_color, color=text_color),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)"
        )
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=grid_color, color=text_color, row=i, col=1)
        fig.update_yaxes(gridcolor=grid_color, color=text_color, row=i, col=1)
    
    return fig

def generate_pdf_report(df, selected_stock, patterns, backtest_result=None, matches=None):
    """
    生成分析报告（文本格式）
    """
    try:
        # 生成文本内容
        report_text = f"""
ZHITU YISHI - Analysis Report
{'='*50}

Stock: {selected_stock}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CURRENT MARKET
{'='*50}
Latest Price: {df.iloc[-1]['close']:.2f}
Change: {(df.iloc[-1]['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close'] * 100:+.2f}%
High: {df.iloc[-1]['high']:.2f}
Low: {df.iloc[-1]['low']:.2f}

PATTERNS DETECTED
{'='*50}
"""
        
        if patterns:
            report_text += f"Total: {len(patterns)} patterns\n\n"
            for name, info in list(patterns.items())[:10]:
                report_text += f"- {name}: {info.get('信号', 'N/A')}\n"
        else:
            report_text += "No patterns detected\n"
        
        report_text += f"""

DISCLAIMER
{'='*50}
This tool is for research only. Not investment advice.
Stock market has risks. Invest with caution.
"""
        
        # 转换为字节
        report_bytes = report_text.encode('utf-8')
        
        # 调试：打印字节长度
        print(f"DEBUG: Report bytes length = {len(report_bytes)}")
        
        if report_bytes and len(report_bytes) > 0:
            return report_bytes
        else:
            print("DEBUG: Report bytes is empty!")
            return None
        
    except Exception as e:
        print(f"DEBUG: Exception in generate_pdf_report: {str(e)}")
        return None

# ===================== 主程序 =====================
def main():
    # 标题
    st.markdown('<p class="main-header">🎯 智图忆市 Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">K线形态量化分析 · 历史相似匹配 · 概率统计决策</p>', unsafe_allow_html=True)
    
    # 免责声明
    st.warning("⚠️  本工具仅供学习复盘研究，不构成任何投资建议。股市有风险，投资需谨慎。")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 分析设置")
        
        # 强制深色模式（固定，不提供切换）
        dark_mode = True
        st.markdown("""
        <style>
            .stApp { background-color: #0e1117 !important; }
            .stMainBlockContainer { background-color: #0e1117 !important; }
            [data-testid="stSidebar"] { background-color: #161b22 !important; }
            .metric-card {background:#161b22 !important; border-left:4px solid #4a90e2 !important; color:#eee !important;}
            .match-card {background:#161b22 !important; border-left:4px solid #ffa500 !important; color:#eee !important;}
            .stButton > button { background-color: #238636 !important; color: #fff !important; border: none !important; }
            .stButton > button:hover { background-color: #2ea043 !important; }
            hr { border-color: #30363d !important; }
        </style>
        """, unsafe_allow_html=True)

        # 标的
        st.subheader("📊 选择标的")
        
        # ===== P0: 自定义股票代码输入 =====
        # 两种模式：下拉选择 或 手动输入，用 radio 切换
        input_mode = st.radio(
            "选股方式",
            ["📋 预设标的", "✏️ 自定义代码"],
            horizontal=True,
            index=0
        )
        
        import re
        if input_mode == "✏️ 自定义代码":
            custom_code_input = st.text_input(
                "股票代码",
                placeholder="如 sh000001 或 sz399006",
                help="sh=上证，sz=深证，后接6位数字",
                key="custom_code_input"
            )
            if custom_code_input:
                clean = custom_code_input.strip().lower()
                if re.match(r"^(sh|sz)\d{6}$", clean):
                    selected = f"自选 {clean.upper()}"
                    code = clean
                else:
                    st.error("⚠️ 格式错误，请输入如 sh000001 或 sz399006")
                    selected = list(STOCK_MAP.keys())[0]
                    code = list(STOCK_MAP.values())[0]
            else:
                st.caption("请输入股票代码后点击开始分析")
                selected = list(STOCK_MAP.keys())[0]
                code = list(STOCK_MAP.values())[0]
        else:
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
        
        # ===== 策略回测设置（放在 expander 外层，保证能跨 render 检测）=====
        BACKTEST_PATTERN_LIST = ["双底", "双顶", "上升三角形", "下降三角形", "头肩顶", "头肩底",
                           "上升旗形", "下降旗形", "上升楔形", "下降楔形", "矩形整理", "杯柄形态",
                           "看涨吞没", "看跌吞没", "早晨之星", "黄昏之星", "锤子线", "上吊线",
                           "射击之星", "看涨孕育线", "看跌孕育线", "三白兵", "三乌鸦",
                           "向上跳空", "向下跳空", "均线多头排列", "均线空头排列", "均线金叉",
                           "均线死叉", "RSI超卖", "RSI超买", "MACD金叉", "MACD死叉",
                           "BOLL上轨突破", "BOLL下轨突破"]
        
        with st.expander("📉 策略回测", expanded=False):
            st.subheader("回测设置")
            # 默认值保留
            st.session_state.setdefault("bt_pattern", BACKTEST_PATTERN_LIST[0])
            st.session_state.setdefault("bt_holding", 20)
            
            default_idx = BACKTEST_PATTERN_LIST.index(st.session_state["bt_pattern"])
            backtest_pattern = st.selectbox(
                "选择回测策略", BACKTEST_PATTERN_LIST,
                index=default_idx,
                key="backtest_pattern_select"
            )
            backtest_holding = st.slider(
                "持有天数", 5, 60,
                value=st.session_state["bt_holding"],
                key="backtest_holding_slider"
            )
            backtest_btn = st.button("🚀 开始回测", key="backtest_btn", use_container_width=True)
        
        # 持久化用户选择（按钮点击后下次 render 仍保留）
        st.session_state["bt_pattern"] = st.session_state.get("backtest_pattern_select", BACKTEST_PATTERN_LIST[0])
        st.session_state["bt_holding"] = st.session_state.get("backtest_holding_slider", 20)
        
        # ===== 回测触发：sidebar 按钮触发 =====
        # backtest_btn 在 expander 内，Streamlit 会自动在 session_state 中置 True
        # 我们在这里统一处理两个回测入口（sidebar 和 main area）
        _backtest_triggered = backtest_btn or st.session_state.get("_backtest_from_main", False)
        _backtest_pattern_val = st.session_state.get("backtest_pattern_select", BACKTEST_PATTERN_LIST[0])
        _backtest_holding_val = st.session_state.get("backtest_holding_slider", 20)
        st.session_state["_backtest_from_main"] = False  # 用完即重置
        
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
        patterns = {}  # 初始化
        if detect_pattern:
            st.markdown("---")
            st.subheader("🔍 当前形态检测")
            
            patterns = detect_patterns(df, window=pattern_days)
            
            # 形态颜色分类
            PATTERN_COLORS = {
                # 经典反转形态 - 红色系（看涨）
                "双底":        {"bg": "#fff0f0", "border": "#e24a4a", "icon": "🔴"},
                "头肩底":      {"bg": "#fff0f0", "border": "#e24a4a", "icon": "🔴"},
                "早晨之星":    {"bg": "#fff0f0", "border": "#e24a4a", "icon": "🔴"},
                "锤子线":      {"bg": "#fff0f0", "border": "#e24a4a", "icon": "🔴"},
                "看涨吞没":    {"bg": "#fff0f0", "border": "#e24a4a", "icon": "🔴"},
                "看涨孕育线":  {"bg": "#fff0f0", "border": "#e24a4a", "icon": "🔴"},
                "三白兵":      {"bg": "#fff0f0", "border": "#e24a4a", "icon": "🔴"},
                "杯柄形态":    {"bg": "#fff0f0", "border": "#e24a4a", "icon": "🔴"},
                # 经典顶部形态 - 绿色系（看跌）
                "双顶":        {"bg": "#f0fff4", "border": "#2e7d32", "icon": "🟢"},
                "头肩顶":      {"bg": "#f0fff4", "border": "#2e7d32", "icon": "🟢"},
                "黄昏之星":    {"bg": "#f0fff4", "border": "#2e7d32", "icon": "🟢"},
                "上吊线":      {"bg": "#f0fff4", "border": "#2e7d32", "icon": "🟢"},
                "射击之星":    {"bg": "#f0fff4", "border": "#2e7d32", "icon": "🟢"},
                "看跌吞没":    {"bg": "#f0fff4", "border": "#2e7d32", "icon": "🟢"},
                "看跌孕育线":  {"bg": "#f0fff4", "border": "#2e7d32", "icon": "🟢"},
                "三乌鸦":      {"bg": "#f0fff4", "border": "#2e7d32", "icon": "🟢"},
                # 整理形态 - 蓝色系（中性）
                "上升三角形":  {"bg": "#f0f4ff", "border": "#1f77b4", "icon": "🔵"},
                "下降三角形":  {"bg": "#f0f4ff", "border": "#1f77b4", "icon": "🔵"},
                "矩形整理":    {"bg": "#f0f4ff", "border": "#1f77b4", "icon": "🔵"},
                "上升旗形":    {"bg": "#f0f4ff", "border": "#1f77b4", "icon": "🔵"},
                "下降旗形":    {"bg": "#f0f4ff", "border": "#1f77b4", "icon": "🔵"},
                "上升楔形":    {"bg": "#f0f4ff", "border": "#1f77b4", "icon": "🔵"},
                "下降楔形":    {"bg": "#f0f4ff", "border": "#1f77b4", "icon": "🔵"},
                # 均线/指标形态 - 橙色系
                "均线多头排列": {"bg": "#fff8f0", "border": "#ff7f0e", "icon": "🟠"},
                "均线空头排列": {"bg": "#fff8f0", "border": "#ff7f0e", "icon": "🟠"},
                "均线金叉":    {"bg": "#fff8f0", "border": "#ff7f0e", "icon": "🟠"},
                "均线死叉":    {"bg": "#fff8f0", "border": "#ff7f0e", "icon": "🟠"},
                "RSI超卖":     {"bg": "#fff8f0", "border": "#ff7f0e", "icon": "🟠"},
                "RSI超买":     {"bg": "#fff8f0", "border": "#ff7f0e", "icon": "🟠"},
                "MACD金叉":    {"bg": "#fff8f0", "border": "#ff7f0e", "icon": "🟠"},
                "MACD死叉":    {"bg": "#fff8f0", "border": "#ff7f0e", "icon": "🟠"},
                "BOLL上轨突破": {"bg": "#fff8f0", "border": "#ff7f0e", "icon": "🟠"},
                "BOLL下轨突破": {"bg": "#fff8f0", "border": "#ff7f0e", "icon": "🟠"},
                "向上跳空":    {"bg": "#fff8f0", "border": "#ff7f0e", "icon": "🟠"},
                "向下跳空":    {"bg": "#fff8f0", "border": "#ff7f0e", "icon": "🟠"},
            }
            
            if patterns:
                cols = st.columns(min(len(patterns), 4))
                for i, (name, info) in enumerate(patterns.items()):
                    with cols[i % 4]:
                        # 获取颜色配置，默认灰色
                        color = PATTERN_COLORS.get(name, {"bg": "transparent", "border": "#888", "icon": "⚪"})
                        signal_text_color = "#3fb950" if "看涨" in info["信号"] else "#f85149" if "看跌" in info["信号"] else "#8b949e"
                        st.markdown(f"""
                        <div style="background:transparent; padding:10px; border-radius:8px;
                                    border-left:4px solid {color['border']}; margin:4px 0;">
                            <div style="font-size:1rem; font-weight:bold; color:#c9d1d9; margin-bottom:4px;">
                                {color['icon']} {name}
                            </div>
                            <div style="color:{signal_text_color}; font-weight:bold; font-size:0.9rem;">
                                {info['信号']}
                            </div>
                            <div style="color:#8b949e; font-size:0.82rem;">
                                置信度: {info['置信度']:.0%} &nbsp;|&nbsp; {info['描述']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("未检测到明确形态")
        
        # ===== 第三部分：形态匹配 =====
        st.markdown("---")
        st.subheader(f"🔎 历史相似形态匹配 (模板: 近 {pattern_days} 日 → 观察: 后 {extend_days} 日)")
        
        matches = []  # 初始化
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
        
        # ===== 第四部分：策略回测 =====
        st.markdown("---")
        st.subheader("📉 策略回测")
        
        col_backtest1, col_backtest2, col_backtest3 = st.columns([2, 1, 1])
        
        with col_backtest1:
            backtest_pattern_select = st.selectbox(
                "选择回测策略",
                ["双底", "双顶", "上升三角形", "下降三角形", "头肩顶", "头肩底", 
                 "上升旗形", "下降旗形", "上升楔形", "下降楔形", "矩形整理", "杯柄形态",
                 "看涨吞没", "看跌吞没", "早晨之星", "黄昏之星", "锤子线", "上吊线",
                 "射击之星", "看涨孕育线", "看跌孕育线", "三白兵", "三乌鸦", 
                 "向上跳空", "向下跳空", "均线多头排列", "均线空头排列", "均线金叉", 
                 "均线死叉", "RSI超卖", "RSI超买", "MACD金叉", "MACD死叉", 
                 "BOLL上轨突破", "BOLL下轨突破"],
                key="backtest_pattern_main"
            )
        
        with col_backtest2:
            backtest_holding_select = st.slider("持有天数", 5, 60, 20, key="backtest_holding_main")
        
        with col_backtest3:
            backtest_run = st.button("🚀 开始回测", key="backtest_run_main", width='stretch')
        
        if backtest_run:
            with st.spinner(f"正在回测 {backtest_pattern_select} 策略..."):
                backtest_result = backtest_pattern_strategy(df, backtest_pattern_select, backtest_holding_select)

            if backtest_result:
                show_backtest_results(backtest_result)
                st.session_state["last_backtest"] = backtest_result
            else:
                st.warning("该形态历史数据不足，无法回测")

        # 每次render都检查，存在就显示回测导出按钮
        if st.session_state.get("backtest_csv_bytes"):
            col_bt_fmt, col_bt_dl = st.columns([2, 1])
            with col_bt_fmt:
                bt_export_fmt = st.selectbox(
                    "回测导出格式",
                    ["📊 CSV 表格", "📄 PDF 报告"],
                    label_visibility="collapsed",
                    key="bt_export_fmt"
                )
            with col_bt_dl:
                if bt_export_fmt == "📄 PDF 报告":
                    # 用reportlab生成简单PDF
                    try:
                        from reportlab.pdfgen import canvas as pdf_canvas
                        from reportlab.lib.pagesizes import letter
                        import io as io_module2
                        buf = io_module2.BytesIO()
                        c = pdf_canvas.Canvas(buf, pagesize=letter)
                        w, h = letter
                        c.setFont("Helvetica-Bold", 16)
                        c.drawString(50, h-50, "Backtest Report - ZhiTu YiShi Pro")
                        c.setFont("Helvetica", 11)
                        y = h - 90
                        bt = st.session_state.get("last_backtest", {})
                        lines = [
                            f"Pattern: {bt.get('pattern_name','')}",
                            f"Holding Days: {bt.get('holding_days','')}",
                            f"Total Trades: {bt.get('total_trades','')}",
                            f"Win Rate: {bt.get('win_rate',0)*100:.1f}%",
                            f"Avg Return: {bt.get('avg_return',0)*100:.2f}%",
                            f"Max Return: {bt.get('max_return',0)*100:.2f}%",
                            f"Sharpe Ratio: {bt.get('sharpe_ratio',0):.2f}",
                        ]
                        for line in lines:
                            c.drawString(50, y, line)
                            y -= 20
                        c.save()
                        buf.seek(0)
                        pdf_data = buf.getvalue()
                        st.download_button(
                            "📥 下载PDF",
                            pdf_data,
                            st.session_state.get("backtest_csv_name","回测结果").replace(".csv",".pdf"),
                            "application/pdf",
                            key="download-backtest-pdf"
                        )
                    except Exception as e:
                        st.error(f"PDF生成失败: {e}")
                else:
                    st.download_button(
                        "📥 下载CSV",
                        st.session_state["backtest_csv_bytes"],
                        st.session_state.get("backtest_csv_name", "回测结果.csv"),
                        "text/csv",
                        key="download-backtest-csv"
                    )
        
        # ===== 第五部分：报告生成 =====
        st.markdown("---")
        st.subheader("📄 生成分析报告")

        col_fmt, col_btn = st.columns([2, 1])
        with col_fmt:
            report_format = st.selectbox(
                "报告格式",
                ["📄 PDF 报告", "📊 CSV 表格", "🌐 HTML 网页"],
                label_visibility="collapsed"
            )
        with col_btn:
            gen_btn = st.button("📄 生成报告", key="generate_report_btn", width='stretch')

        if gen_btn:
            backtest_for_report = st.session_state.get("last_backtest", None)
            report_bytes = generate_pdf_report(df, selected, patterns, backtest_for_report, matches)
            if report_bytes:
                # 根据格式转换内容
                if report_format == "🌐 HTML 网页":
                    # 转为HTML格式
                    text_content = report_bytes.decode('utf-8')
                    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>智图忆市分析报告 - {selected}</title>
<style>
body{{font-family:Arial,sans-serif;max-width:800px;margin:40px auto;padding:20px;background:#f9f9f9;}}
h1{{color:#1f77b4;}} pre{{background:#fff;padding:20px;border-radius:8px;border:1px solid #ddd;white-space:pre-wrap;}}
</style></head>
<body><h1>📊 智图忆市分析报告</h1><pre>{text_content}</pre></body></html>"""
                    final_bytes = html_content.encode('utf-8')
                    ext, mime = "html", "text/html"
                elif report_format == "📊 CSV 表格":
                    # 转为CSV格式（提取关键数据）
                    lines = report_bytes.decode('utf-8').split('\n')
                    csv_lines = ["项目,内容"]
                    for line in lines:
                        if ':' in line and not line.startswith('='):
                            parts = line.split(':', 1)
                            csv_lines.append(f'"{parts[0].strip()}","{parts[1].strip()}"')
                    final_bytes = '\n'.join(csv_lines).encode('utf-8-sig')
                    ext, mime = "csv", "text/csv"
                else:
                    # PDF格式
                    try:
                        from reportlab.pdfgen import canvas as pdf_canvas
                        from reportlab.lib.pagesizes import letter
                        import io as io_module2
                        buf = io_module2.BytesIO()
                        c = pdf_canvas.Canvas(buf, pagesize=letter)
                        w, h = letter
                        c.setFont("Helvetica-Bold", 18)
                        c.drawString(50, h-50, "ZhiTu YiShi Pro - Analysis Report")
                        c.setFont("Helvetica", 11)
                        y = h - 90
                        for line in report_bytes.decode("utf-8").split("\n")[:40]:
                            if y < 60:
                                c.showPage()
                                y = h - 50
                                c.setFont("Helvetica", 11)
                            c.drawString(50, y, line[:90])
                            y -= 16
                        c.save()
                        buf.seek(0)
                        final_bytes = buf.getvalue()
                    except Exception:
                        final_bytes = report_bytes
                    ext, mime = "pdf", "application/pdf"

                fname = f"智图忆市分析报告_{selected}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
                st.session_state["report_bytes"] = final_bytes
                st.session_state["report_name"] = fname
                st.session_state["report_mime"] = mime
                st.success("✅ 报告生成成功！")
            else:
                st.error("❌ 报告生成失败")

        # 每次render都检查，存在就显示下载按钮
        if st.session_state.get("report_bytes"):
            st.download_button(
                "📥 下载报告",
                st.session_state["report_bytes"],
                st.session_state.get("report_name", "报告.txt"),
                st.session_state.get("report_mime", "text/plain"),
                key="download-report"
            )
    
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
