import pandas as pd
import numpy as np

def find_optimal_threshold(df, stock_code, start_date, end_date, indicator, higher_is_buy=True):
    """
    找出给定股票在指定时间段内，情绪指标的最优分位数阈值
    
    参数:
    df (DataFrame): 包含股票数据的DataFrame
    stock_code (str): 股票代码
    start_date (str): 开始日期，格式为'YYYY-MM-DD'
    end_date (str): 结束日期，格式为'YYYY-MM-DD'
    indicator (str): 情绪指标的列名
    higher_is_buy (bool): 如果为True，指标高于阈值时买入；如果为False，指标低于阈值时买入
    
    返回:
    dict: 包含最优分位数、阈值和收益率的字典
    """
    # 筛选特定股票的数据
    stock_df = df.query(f'股票代码 == "{stock_code}"').copy()
    
    # 确保日期格式正确并排序
    stock_df['日期'] = pd.to_datetime(stock_df['日期'])
    stock_df = stock_df.sort_values('日期')
    
    # 计算每日收益率（用于后续计算策略收益）
    stock_df['daily_return'] = stock_df['close'].pct_change()
    
    # 筛选时间范围内的数据
    mask = (stock_df['日期'] >= pd.to_datetime(start_date)) & (stock_df['日期'] <= pd.to_datetime(end_date))
    backtest_df = stock_df[mask].copy()
    
    # 如果回测数据为空，返回错误信息
    if len(backtest_df) == 0:
        return {"error": "No data available for the specified time period and stock code"}
    
    # 计算全年数据的分位数（0%到100%，步长为1%）
    percentiles = np.arange(0, 101, 1)
    thresholds = np.percentile(stock_df[indicator], percentiles)
    
    # 存储每个阈值的回测结果
    results = []
    
    # 对每个阈值进行回测
    for p, threshold in zip(percentiles, thresholds):
        # 生成信号：根据higher_is_buy参数决定买入条件
        if higher_is_buy:
            backtest_df['signal'] = (backtest_df[indicator] > threshold).astype(int)
        else:
            backtest_df['signal'] = (backtest_df[indicator] < threshold).astype(int)
        
        # 计算持仓：信号的累积最大值（一旦买入就持有，直到卖出信号出现）
        backtest_df['position'] = backtest_df['signal'].copy()
        
        # 计算前一天的持仓，用于确定交易
        backtest_df['prev_position'] = backtest_df['position'].shift(1).fillna(0)
        
        # 计算策略收益率：只有在持仓时才能获得收益
        backtest_df['strategy_return'] = backtest_df['daily_return'] * backtest_df['prev_position']
        
        # 计算累积收益率
        cumulative_return = (1 + backtest_df['strategy_return'].fillna(0)).prod() - 1
        
        results.append({
            'percentile': p,
            'threshold': threshold,
            'return': cumulative_return
        })
    
    # 找出收益率最高的结果
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df['return'].idxmax()]
    
    return {
        'optimal_percentile': best_result['percentile'],
        'optimal_threshold': best_result['threshold'],
        'max_return': best_result['return']
    }
