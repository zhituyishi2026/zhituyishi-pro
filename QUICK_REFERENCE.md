# 智图忆市 v2.0 快速参考

## 📊 26种形态检测速查表

| # | 形态名称 | 英文名 | 信号 | 置信度 |
|---|---------|--------|------|--------|
| 1 | 双底 | Double Bottom | 看涨 | 85% |
| 2 | 双顶 | Double Top | 看跌 | 85% |
| 3 | 上升三角形 | Rising Triangle | 看涨 | 75% |
| 4 | 下降三角形 | Falling Triangle | 看跌 | 75% |
| 5 | 头肩顶 | Head & Shoulders Top | 看跌 | 78% |
| 6 | 头肩底 | Head & Shoulders Bottom | 看涨 | 78% |
| 7 | 上升旗形 | Bull Flag | 看涨 | 72% |
| 8 | 下降旗形 | Bear Flag | 看跌 | 72% |
| 9 | 上升楔形 | Rising Wedge | 看跌 | 68% |
| 10 | 下降楔形 | Falling Wedge | 看涨 | 68% |
| 11 | 矩形整理 | Rectangle | 中性 | 70% |
| 12 | 杯柄形态 | Cup and Handle | 看涨 | 75% |
| 13 | 看涨吞没 | Bullish Engulfing | 看涨 | 76% |
| 14 | 看跌吞没 | Bearish Engulfing | 看跌 | 76% |
| 15 | 早晨之星 | Morning Star | 看涨 | 74% |
| 16 | 黄昏之星 | Evening Star | 看跌 | 74% |
| 17 | 锤子线 | Hammer | 看涨 | 72% |
| 18 | 上吊线 | Hanging Man | 看跌 | 72% |
| 19 | 射击之星 | Shooting Star | 看跌 | 71% |
| 20 | 看涨孕育线 | Bullish Harami | 看涨 | 68% |
| 21 | 看跌孕育线 | Bearish Harami | 看跌 | 68% |
| 22 | 三白兵 | Three White Soldiers | 看涨 | 77% |
| 23 | 三乌鸦 | Three Black Crows | 看跌 | 77% |
| 24 | 向上跳空 | Gap Up | 看涨 | 65% |
| 25 | 向下跳空 | Gap Down | 看跌 | 65% |
| 26 | BOLL突破 | Bollinger Bands | 看涨/看跌 | 68% |

## 🔧 主要函数速查

### 形态检测
```python
patterns = detect_patterns(df, window=20)
# 返回: {形态名: {置信度, 描述, 信号}}
```

### 形态匹配
```python
matches = find_similar_patterns(df, pattern_days=20, top_n=10)
# 返回: [{相似度, 日期, 收益率, ...}]
```

### 策略回测
```python
result = backtest_pattern_strategy(df, pattern_name="双底", holding_days=20)
# 返回: {交易次数, 胜率, 平均收益, 夏普比率, ...}

show_backtest_results(result)  # 显示回测结果UI
```

### PDF报告生成
```python
pdf_bytes = generate_pdf_report(df, selected_stock, patterns, backtest_result, matches)
# 返回: PDF字节流，可用 st.download_button 下载
```

## 📈 UI组件位置

### 侧边栏
- 📊 选择标的
- 🔧 形态匹配设置
- 📐 图表设置
- 🔍 形态检测
- 📉 策略回测（可折叠）

### 主界面
- 📌 第一部分：当前行情
- 🔍 第二部分：形态检测
- 🔎 第三部分：形态匹配
- 📉 第四部分：策略回测
- 📄 第五部分：PDF报告生成

## 💡 使用技巧

### 形态匹配
- 模板窗口天数：5-20天用于短期，20-60天用于中期
- 观察天数：与持有期对应，通常设置为20-30天
- 匹配数量：10-15个最优，过多会降低相似度

### 策略回测
- 持有天数：5天（超短期）、10天（短期）、20天（中期）、30天（长期）
- 胜率 > 50% 表示策略有效
- 夏普比率 > 1 表示风险调整后收益良好
- 最大亏损反映最坏情况

### PDF报告
- 包含完整的分析结果和免责声明
- 适合分享给他人或存档
- 文件名自动包含标的和时间戳

## ⚠️ 注意事项

1. **数据质量**：确保数据源（新浪/AKShare）可用
2. **历史数据**：至少需要300天数据进行有效匹配
3. **形态置信度**：置信度仅供参考，不保证准确性
4. **回测局限**：历史表现不代表未来结果
5. **风险提示**：本工具仅供学习研究，不构成投资建议

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行应用
streamlit run app.py

# 3. 在浏览器中打开
# http://localhost:8501
```

## 📊 数据源

- **新浪财经**：主要数据源，日K线数据
- **AKShare**：备选数据源，确保数据可用性

## 🔄 更新日志

### v2.0 (2026-03-29)
- ✨ 扩充形态检测库到26种
- ✨ 新增策略回测模块
- ✨ 新增PDF报告生成
- 🐛 优化代码结构和性能
- 📝 完善中文注释和文档

### v1.0 (初始版本)
- 基础形态匹配功能
- 10种形态检测
- 历史相似形态查询

---

**需要帮助？查看 UPGRADE_SUMMARY.md 获取详细信息。**
