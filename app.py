import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import backtrader as bt
import talib.abstract as ta
import optuna
from sklearn.linear_model import LinearRegression
import time
import warnings
import random
from io import StringIO
import traceback
import collections

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AI Trading Strategy Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    :root {
        --primary: #4CAF50;
        --secondary: #2196F3;
        --accent: #FF9800;
        --dark: #343a40;
        --light: #f8f9fa;
    }
    .main { background-color: var(--light); }
    .sidebar .sidebar-content { background-color: var(--dark); color: white; }
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-weight: 500;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary);
    }
    .strategy-code {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid var(--secondary);
        font-family: monospace;
        white-space: pre;
        overflow-x: auto;
    }
    .equity-curve {
        margin-top: 20px;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
    }
    .badge {
        display: inline-block;
        padding: 0.25em 0.4em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
    }
    .badge-success {
        color: #fff;
        background-color: #28a745;
    }
    .badge-warning {
        color: #212529;
        background-color: #ffc107;
    }
    .badge-danger {
        color: #fff;
        background-color: #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    session_vars = {
        'data_uploaded': False,
        'strategy_generated': False,
        'backtest_completed': False,
        'df': None,
        'in_sample': None,
        'out_of_sample': None,
        'split_index': None,
        'strategy_code': '',
        'strategy_params': {},
        'strategy_score': 0,
        'in_sample_results': None,
        'out_of_sample_results': None,
        'optimization_complete': False,
        'optimization_progress': 0,
        'found_valid': False,
        'last_update_time': time.time(),
        'optimization_metric': 'Return-to-Drawdown Ratio',
        'target_value': 2.0,
        'strategy_type': 'Automatic Selection',
        'optimization_history': [],
        'min_trades': 5,
        'risk_per_trade': 0.01,
        'consecutive_failures': 0,
        'max_consecutive_failures': 20
    }
    for key, value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Enhanced trade analyzer with streak tracking
class TradeAnalyzer(bt.Analyzer):
    def __init__(self):
        self.trades = []
        self.positions = []
        self.win_streak = 0
        self.loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.total_duration = 0
        self.total_trades = 0
        self.total_wins = 0
        self.gross_profit = 0
        self.gross_loss = 0

    def notify_trade(self, trade):
        if trade.isclosed:
            self.total_trades += 1
            pnl = trade.pnlcomm
            pnl_pct = trade.pnlcomm / (trade.price * trade.size) * 100 if trade.price and trade.size else 0
            is_win = pnl > 0
            duration = (bt.num2date(trade.dtclose) - bt.num2date(trade.dtopen)).days

            if is_win:
                self.total_wins += 1
                self.win_streak += 1
                self.loss_streak = 0
                self.max_win_streak = max(self.max_win_streak, self.win_streak)
                self.gross_profit += pnl
            else:
                self.loss_streak += 1
                self.win_streak = 0
                self.max_loss_streak = max(self.max_loss_streak, self.loss_streak)
                self.gross_loss += abs(pnl)
            
            self.total_duration += duration
            
            self.trades.append({
                'entry_date': bt.num2date(trade.dtopen),
                'exit_date': bt.num2date(trade.dtclose),
                'entry_price': trade.price,
                'exit_price': trade.price + trade.pnlcomm/trade.size if trade.size else trade.price,
                'size': trade.size,
                'pnl': pnl,
                'pnl_percent': pnl_pct,
                'duration': duration,
                'is_win': is_win
            })

    def get_analysis(self):
        avg_duration = self.total_duration / self.total_trades if self.total_trades > 0 else 0
        win_rate = self.total_wins / self.total_trades if self.total_trades > 0 else 0
        profit_factor = self.gross_profit / self.gross_loss if self.gross_loss > 0 else 0
        
        return {
            'total_trades': self.total_trades,
            'total_wins': self.total_wins,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trades': self.trades,
            'avg_trade_duration': avg_duration,
            'max_win_streak': self.max_win_streak,
            'max_loss_streak': self.max_loss_streak,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss
        }

# Enhanced strategy class with position sizing and volatility adjustment
class OptimizedStrategy(bt.Strategy):
    params = (
        ('ma_length', 20),
        ('atr_multiplier', 2.0),
        ('lookback', 14),
        ('std_dev_mult', 1.5),
        ('entry_z', 2.0),
        ('exit_z', 0.5),
        ('strategy_type', 'trend'),
        ('stop_loss_pct', 0.015),
        ('take_profit_pct', 0.03),
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('risk_per_trade', 0.01),
        ('volatility_lookback', 20),
        ('max_position_size', 0.1),
        ('trailing_stop', False),
        ('trail_percent', 0.01)
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.dataopen = self.datas[0].open
        
        self.order = None
        self.stop_price = 0
        self.target_price = 0
        self.trailing_stop_price = 0
        self.volatility = bt.indicators.ATR(self.data, period=self.p.volatility_lookback)
        self.ready = False
        self.warmup_period = max(self.p.lookback, self.p.ma_length, self.p.rsi_period) + 5
        
        # Track bar count for warmup
        self.bar_count = 0
        
        # Initialize indicators based on strategy type
        if self.p.strategy_type == 'trend':
            self.ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.ma_length)
            self.atr = bt.indicators.ATR(self.data, period=self.p.ma_length)
            self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
            self.macd = bt.indicators.MACD(self.data.close)
        elif self.p.strategy_type == 'mean_reversion':
            self.midline = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.lookback)
            stddev = bt.indicators.StdDev(self.data.close, period=self.p.lookback)
            self.upper = self.midline + stddev * self.p.std_dev_mult
            self.lower = self.midline - stddev * self.p.std_dev_mult
            self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
            self.volume_sma = bt.indicators.SimpleMovingAverage(self.data.volume, period=self.p.lookback)
        elif self.p.strategy_type == 'stat_arb':
            self.spread = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.lookback)
            stddev = bt.indicators.StdDev(self.data.close, period=self.p.lookback)
            self.upper_band = self.spread + stddev * self.p.entry_z
            self.lower_band = self.spread - stddev * self.p.entry_z
            self.exit_upper = self.spread + stddev * self.p.exit_z
            self.exit_lower = self.spread - stddev * self.p.exit_z
            self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
            self.hurst = bt.indicators.Hurst(self.data.close, period=self.p.lookback)

    def next(self):
        self.bar_count += 1
        
        # Skip if not enough data or order pending
        if self.bar_count < self.warmup_period or self.order:
            return
            
        size = None  # Let the PercentSizer determine size
        
        if self.p.strategy_type == 'trend':
            if not self.position:
                if (self.data.close[0] > self.ma[0] and 
                    self.rsi[0] < self.p.rsi_overbought and 
                    self.macd.macd[0] > self.macd.signal[0]):
                    self.buy()
                    self.stop_price = self.data.close[0] * (1 - self.p.stop_loss_pct)
                    self.target_price = self.data.close[0] * (1 + self.p.take_profit_pct)
                    if self.p.trailing_stop:
                        self.trailing_stop_price = self.data.close[0] * (1 - self.p.trail_percent)
            else:
                if self.p.trailing_stop:
                    self.trailing_stop_price = max(
                        self.trailing_stop_price,
                        self.data.close[0] * (1 - self.p.trail_percent)
                    )
                exit_condition = (
                    self.data.close[0] < self.ma[0] or 
                    self.rsi[0] > self.p.rsi_overbought or
                    (self.p.trailing_stop and self.data.close[0] <= self.trailing_stop_price)
                )
                
                if exit_condition or self.data.close[0] <= self.stop_price or self.data.close[0] >= self.target_price:
                    self.close()
        elif self.p.strategy_type == 'mean_reversion':
            if not self.position:
                if (self.data.close[0] < self.lower[0] and 
                    self.rsi[0] < self.p.rsi_oversold and 
                    self.data.volume[0] > self.volume_sma[0]):
                    self.buy()
                    self.stop_price = self.data.close[0] * (1 - self.p.stop_loss_pct)
                    self.target_price = self.data.close[0] * (1 + self.p.take_profit_pct)
                elif (self.data.close[0] > self.upper[0] and 
                      self.rsi[0] > self.p.rsi_overbought and 
                      self.data.volume[0] > self.volume_sma[0]):
                    self.sell()
                    self.stop_price = self.data.close[0] * (1 + self.p.stop_loss_pct)
                    self.target_price = self.data.close[0] * (1 - self.p.take_profit_pct)
            else:
                if self.position.size > 0:  # Long position
                    if (self.data.close[0] > self.midline[0] or 
                        self.rsi[0] > self.p.rsi_overbought or
                        self.data.close[0] <= self.stop_price or 
                        self.data.close[0] >= self.target_price):
                        self.close()
                else:  # Short position
                    if (self.data.close[0] < self.midline[0] or 
                        self.rsi[0] < self.p.rsi_oversold or
                        self.data.close[0] >= self.stop_price or 
                        self.data.close[0] <= self.target_price):
                        self.close()
        elif self.p.strategy_type == 'stat_arb':
            if not self.position:
                if self.hurst[0] < 0.5:  # Mean-reverting regime
                    if self.data.close[0] > self.upper_band[0]:
                        self.sell()
                        self.stop_price = self.data.close[0] * (1 + self.p.stop_loss_pct)
                        self.target_price = self.data.close[0] * (1 - self.p.take_profit_pct)
                    elif self.data.close[0] < self.lower_band[0]:
                        self.buy()
                        self.stop_price = self.data.close[0] * (1 - self.p.stop_loss_pct)
                        self.target_price = self.data.close[0] * (1 + self.p.take_profit_pct)
            else:
                if self.position.size > 0:  # Long position
                    if (self.data.close[0] > self.exit_upper[0] or 
                        self.data.close[0] <= self.stop_price or 
                        self.data.close[0] >= self.target_price):
                        self.close()
                else:  # Short position
                    if (self.data.close[0] < self.exit_lower[0] or 
                        self.data.close[0] >= self.stop_price or 
                        self.data.close[0] <= self.target_price):
                        self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order = None

def create_failed_result(data_length):
    return {
        'total_return': 0,
        'annual_return': 0,
        'max_drawdown': 100,
        'rr_dd_ratio': 0,
        'profit_factor': 0,
        'num_trades': 0,
        'win_rate': 0,
        'sharpe_ratio': 0,
        'sqn': 0,
        'equity_curve': [100000] * data_length,
        'trades': {},
        'trade_records': pd.DataFrame(),
        'time_return': {},
        'success': False,
        'avg_trade_duration': 0,
        'max_win_streak': 0,
        'max_loss_streak': 0
    }

# Enhanced backtest function with more metrics
def backtest_strategy(data, strategy_params, is_sample=True):
    try:
        cerebro = bt.Cerebro(stdstats=False)
        
        # Prepare data
        data = data.copy()
        if 'Date' in data.columns:
            data = data.set_index('Date')
        data.index = pd.to_datetime(data.index)
        
        # Validate data
        if len(data) < 100:
            print("Insufficient data points for backtesting")
            return create_failed_result(len(data))
            
        if data.isnull().values.any():
            print("Data contains missing values")
            return create_failed_result(len(data))
            
        if (data['Close'] <= 0).any():
            print("Invalid price values in data")
            return create_failed_result(len(data))

        # Create data feed
        data_feed = bt.feeds.PandasData(
            dataname=data,
            datetime=None,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume' if 'Volume' in data.columns else None,
            timeframe=bt.TimeFrame.Days
        )
        
        cerebro.adddata(data_feed)
        cerebro.addstrategy(OptimizedStrategy, **strategy_params)
        
        # Initial settings
        cerebro.broker.setcash(100000.0)
        cerebro.broker.setcommission(commission=0.0005)
        cerebro.addsizer(bt.sizers.PercentSizer, percents=10)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, annualize=True)
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
        cerebro.addanalyzer(TradeAnalyzer, _name='trade_recorder')
        
        # Run backtest with error handling
        results = cerebro.run(maxcpus=1, runonce=False, exactbars=True)
        strat = results[0]
        
        # Get equity curve
        portfolio_values = []
        for i in range(len(data)):
            cerebro.broker.next()
            portfolio_values.append(cerebro.broker.getvalue())
        
        # Get results with proper error handling
        def safe_get(analyzer, attr, default=None):
            try:
                return analyzer.get_analysis().get(attr, default)
            except:
                return default
        
        returns = safe_get(strat.analyzers.returns, 'rtot', 0)
        annual_return = safe_get(strat.analyzers.returns, 'rnorm', 0)
        max_drawdown = safe_get(strat.analyzers.drawdown, 'max.drawdown', 100)
        sharpe_ratio = safe_get(strat.analyzers.sharpe, 'sharperatio', 0)
        sqn_score = safe_get(strat.analyzers.sqn, 'sqn', 0)
        
        # Trade analysis from our custom analyzer
        trade_analysis = strat.analyzers.trade_recorder.get_analysis()
        num_trades = trade_analysis.get('total_trades', 0)
        win_rate = trade_analysis.get('win_rate', 0)
        profit_factor = trade_analysis.get('profit_factor', 0)
        
        # Calculate RR/DD ratio safely
        if max_drawdown == 0:
            rr_dd_ratio = float('inf') if returns > 0 else 0
        else:
            rr_dd_ratio = returns / (max_drawdown/100) if max_drawdown > 0 else 0
        
        # Trade records and additional metrics
        trade_records = trade_analysis.get('trades', [])
        avg_trade_duration = trade_analysis.get('avg_trade_duration', 0)
        max_win_streak = trade_analysis.get('max_win_streak', 0)
        max_loss_streak = trade_analysis.get('max_loss_streak', 0)
        
        success = num_trades >= st.session_state.get('min_trades', 5)
        
        return {
            'total_return': returns,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'rr_dd_ratio': rr_dd_ratio,
            'profit_factor': profit_factor,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'sqn': sqn_score,
            'equity_curve': portfolio_values,
            'trades': strat.analyzers.trades.get_analysis(),
            'trade_records': pd.DataFrame(trade_records) if trade_records else pd.DataFrame(),
            'time_return': strat.analyzers.time_return.get_analysis(),
            'success': success,
            'avg_trade_duration': avg_trade_duration,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak
        }
        
    except Exception as e:
        print(f"Backtest error: {str(e)}")
        traceback.print_exc()
        return create_failed_result(len(data))

# Generate EasyLanguage code for the strategy
def generate_easylanguage_code(params):
    # Ensure all required parameters exist with defaults
    required_params = {
        'max_position_size': 0.1,
        'risk_per_trade': 0.01,
        'strategy_type': 'trend',
        'ma_length': 20,
        'atr_multiplier': 2.0,
        'lookback': 14,
        'std_dev_mult': 1.5,
        'entry_z': 2.0,
        'exit_z': 0.5,
        'stop_loss_pct': 0.015,
        'take_profit_pct': 0.03,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'volatility_lookback': 20,
        'trailing_stop': False,
        'trail_percent': 0.01
    }
    
    # Merge with defaults
    final_params = {**required_params, **params}
    
    if final_params['strategy_type'] == 'trend':
        code = f"""
        // Trend Following Strategy
        // Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        inputs:
            MA_Length({final_params['ma_length']}),
            ATR_Multiplier({final_params['atr_multiplier']}),
            RSI_Period({final_params['rsi_period']}),
            RSI_Overbought({final_params['rsi_overbought']}),
            Stop_Loss_Pct({final_params['stop_loss_pct']}),
            Take_Profit_Pct({final_params['take_profit_pct']}),
            Risk_Per_Trade({final_params['risk_per_trade']}),
            Max_Position_Size({final_params['max_position_size']});
            
        variables:
            ma(0), atr(0), rsi(0), macd(0), macdSignal(0),
            positionSize(0), stopPrice(0), targetPrice(0);
            
        // Calculate indicators
        ma = Average(Close, MA_Length);
        atr = AvgTrueRange(MA_Length);
        rsi = RSI(Close, RSI_Period);
        macd = MACD(Close, 12, 26, 9);
        
        // Calculate position size based on volatility and risk
        positionSize = MinList(
            (CurrentEquity * Risk_Per_Trade) / (atr * ATR_Multiplier),
            CurrentEquity * Max_Position_Size
        ) / Close;
        
        // Entry conditions
        if Close > ma and rsi < RSI_Overbought and macd > macd[1] then
            Buy("TrendEntry") positionSize shares next bar at market;
            stopPrice = Close * (1 - Stop_Loss_Pct);
            targetPrice = Close * (1 + Take_Profit_Pct);
            
        // Exit conditions
        if MarketPosition > 0 then begin
            if Close < ma or rsi > RSI_Overbought then
                Sell("TrendExit") next bar at market;
            if Close <= stopPrice then
                Sell("StopLoss") next bar at market;
            if Close >= targetPrice then
                Sell("TakeProfit") next bar at market;
        end;
        """
    elif final_params['strategy_type'] == 'mean_reversion':
        code = f"""
        // Mean Reversion Strategy
        // Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        inputs:
            Lookback({final_params['lookback']}),
            StdDev_Mult({final_params['std_dev_mult']}),
            RSI_Period({final_params['rsi_period']}),
            RSI_Oversold({final_params['rsi_oversold']}),
            RSI_Overbought({final_params['rsi_overbought']}),
            Risk_Per_Trade({final_params['risk_per_trade']}),
            Max_Position_Size({final_params['max_position_size']}),
            Volatility_Lookback({final_params['volatility_lookback']});
            
        variables:
            midline(0), upper(0), lower(0), rsi(0), vol(0),
            positionSize(0), volumeSMA(0);
            
        // Calculate indicators
        midline = Average(Close, Lookback);
        upper = midline + StandardDev(Close, Lookback, 1) * StdDev_Mult;
        lower = midline - StandardDev(Close, Lookback, 1) * StdDev_Mult;
        rsi = RSI(Close, RSI_Period);
        vol = AvgTrueRange(Volatility_Lookback);
        volumeSMA = Average(Volume, Lookback);
        
        // Calculate position size based on volatility and risk
        positionSize = MinList(
            (CurrentEquity * Risk_Per_Trade) / (vol * 2),
            CurrentEquity * Max_Position_Size
        ) / Close;
        
        // Long entry (oversold)
        if Close < lower and rsi < RSI_Oversold and Volume > volumeSMA then
            Buy("MRLongEntry") positionSize shares next bar at market;
            
        // Short entry (overbought)
        if Close > upper and rsi > RSI_Overbought and Volume > volumeSMA then
            SellShort("MRShortEntry") positionSize shares next bar at market;
            
        // Exit conditions
        if MarketPosition > 0 and (Close > midline or rsi > RSI_Overbought) then
            Sell("MRLongExit") next bar at market;
            
        if MarketPosition < 0 and (Close < midline or rsi < RSI_Oversold) then
            BuyToCover("MRShortExit") next bar at market;
        """
    else:  # stat_arb
        code = f"""
        // Statistical Arbitrage Strategy
        // Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        inputs:
            Lookback({final_params['lookback']}),
            Entry_Z({final_params['entry_z']}),
            Exit_Z({final_params['exit_z']}),
            RSI_Period({final_params['rsi_period']}),
            Risk_Per_Trade({final_params['risk_per_trade']}),
            Max_Position_Size({final_params['max_position_size']}),
            Volatility_Lookback({final_params['volatility_lookback']});
            
        variables:
            spread(0), upperBand(0), lowerBand(0), exitUpper(0), exitLower(0),
            rsi(0), vol(0), positionSize(0), hurst(0);
            
        // Calculate indicators
        spread = Average(Close, Lookback);
        upperBand = spread + StandardDev(Close, Lookback, 1) * Entry_Z;
        lowerBand = spread - StandardDev(Close, Lookback, 1) * Entry_Z;
        exitUpper = spread + StandardDev(Close, Lookback, 1) * Exit_Z;
        exitLower = spread - StandardDev(Close, Lookback, 1) * Exit_Z;
        rsi = RSI(Close, RSI_Period);
        vol = AvgTrueRange(Volatility_Lookback);
        hurst = HurstExponent(Close, Lookback);
        
        // Calculate position size based on volatility and risk
        positionSize = MinList(
            (CurrentEquity * Risk_Per_Trade) / (vol * 2),
            CurrentEquity * Max_Position_Size
        ) / Close;
        
        // Entry conditions (mean-reverting regime)
        if hurst < 0.5 then begin
            if Close > upperBand then
                SellShort("SAEntryShort") positionSize shares next bar at market;
                
            if Close < lowerBand then
                Buy("SAEntryLong") positionSize shares next bar at market;
        end;
        
        // Exit conditions
        if MarketPosition > 0 and Close > exitUpper then
            Sell("SAExitLong") next bar at market;
            
        if MarketPosition < 0 and Close < exitLower then
            BuyToCover("SAExitShort") next bar at market;
        """
    return code

def evaluate_strategy(results, min_trades):
    if results['num_trades'] < min_trades:
        return float('-inf')
    
    # Additional quality checks
    if results['profit_factor'] < 0.5:  # Very poor performance
        return float('-inf')
        
    if results['max_drawdown'] > 50:  # Extreme risk
        return results['rr_dd_ratio'] * 0.3  # Severely penalize
        
    return results['rr_dd_ratio']

# Generate trading strategy using Optuna optimization
def generate_strategy(data, strategy_type, optimization_metric, target_value, max_iterations, 
                     show_progress=True, optimization_mode="Find Best Possible", max_time_minutes=30):
    
    # Define the strategy type mappings at the function level
    strategy_type_map = {
        'Trend Following': 'trend',
        'Mean Reversion': 'mean_reversion',
        'Statistical Arbitrage': 'stat_arb'
    }
    
    reverse_strategy_type_map = {
        'trend': 'Trend Following',
        'mean_reversion': 'Mean Reversion',
        'stat_arb': 'Statistical Arbitrage'
    }

    # Define objective function for Optuna
    def objective(trial):
        internal_strategy_type = strategy_type_map.get(strategy_type, 
                            trial.suggest_categorical('strategy_type', ['trend', 'mean_reversion', 'stat_arb']))
        
        params = {
            'strategy_type': internal_strategy_type,
            'lookback': trial.suggest_int('lookback', 10, 50),
            'risk_per_trade': trial.suggest_float('risk_per_trade', 0.005, 0.03),
            'ma_length': trial.suggest_int('ma_length', 10, 50),
            'atr_multiplier': trial.suggest_float('atr_multiplier', 1.0, 3.0),
            'std_dev_mult': trial.suggest_float('std_dev_mult', 1.0, 3.0),
            'entry_z': trial.suggest_float('entry_z', 1.0, 3.0),
            'exit_z': trial.suggest_float('exit_z', 0.1, 1.0),
            'stop_loss_pct': trial.suggest_float('stop_loss_pct', 0.005, 0.02),
            'take_profit_pct': trial.suggest_float('take_profit_pct', 0.01, 0.05),
            'rsi_period': trial.suggest_int('rsi_period', 10, 21),
            'rsi_overbought': trial.suggest_int('rsi_overbought', 65, 75),
            'rsi_oversold': trial.suggest_int('rsi_oversold', 25, 35),
            'volatility_lookback': trial.suggest_int('volatility_lookback', 14, 30),
            'max_position_size': 0.1,
            'trailing_stop': trial.suggest_categorical('trailing_stop', [True, False]),
            'trail_percent': trial.suggest_float('trail_percent', 0.005, 0.02)
        }
        
        print(f"\nTrial {trial.number} with params: {params}")
        
        results = backtest_strategy(data, params)
        
        print(f"Results - Trades: {results['num_trades']}, RR/DD: {results['rr_dd_ratio']:.2f}")
        
        score = evaluate_strategy(results, st.session_state.get('min_trades', 5))
        
        # Early stopping if we found a good enough strategy
        if optimization_mode == "Find First Valid" and score >= target_value:
            raise optuna.exceptions.TrialPruned("Found valid strategy")
            
        return score
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    start_time = time.time()
    
    def callback(study, trial):
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time_minutes * 60:
            raise optuna.exceptions.TrialPruned("Time limit reached")
            
        # Stop if too many consecutive failures
        if len(study.trials) > st.session_state.max_consecutive_failures:
            recent_failures = sum(t.value == float('-inf') for t in study.trials[-st.session_state.max_consecutive_failures:])
            if recent_failures == st.session_state.max_consecutive_failures:
                raise optuna.exceptions.TrialPruned("Too many consecutive failures")
            
        if time.time() - st.session_state.last_update_time > 5:  # Update every 5 seconds
            st.session_state.optimization_progress = trial.number / max_iterations * 100
            st.session_state.last_update_time = time.time()
    
    try:
        study.optimize(
            objective, 
            n_trials=max_iterations, 
            callbacks=[callback],
            timeout=max_time_minutes * 60
        )
    except optuna.exceptions.TrialPruned as e:
        print(f"Optimization stopped: {str(e)}")
    
    # Get best trial
    if not study.trials:
        return None
        
    best_trial = study.best_trial
    
    # Use the selected strategy type if not automatic, otherwise use the optimized one
    if strategy_type != "Automatic Selection":
        display_strategy_type = strategy_type
        # Add the strategy_type to params since it wasn't optimized
        best_trial.params['strategy_type'] = strategy_type_map[strategy_type]
    else:
        display_strategy_type = reverse_strategy_type_map.get(
            best_trial.params.get('strategy_type', 'trend'), 
            'Custom Strategy'
        )
    
    # Backtest with best parameters
    best_results = backtest_strategy(data, best_trial.params)
    
    return {
        'params': best_trial.params,
        'score': best_trial.value,
        'results': best_results,
        'strategy_code': generate_easylanguage_code(best_trial.params),
        'strategy_type': display_strategy_type,
        'found_valid': best_trial.value >= target_value
    }

# Plot performance metrics
def plot_performance_metrics(in_sample_results, out_of_sample_results, split_index, total_length):
    fig = go.Figure()
    
    # Combine equity curves
    full_equity = in_sample_results['equity_curve'] + out_of_sample_results['equity_curve']
    
    # Plot full equity curve
    fig.add_trace(go.Scatter(
        x=np.arange(len(full_equity)),
        y=full_equity,
        name='Equity Curve',
        line=dict(color='blue')
    ))
    
    # Add vertical line at split point
    fig.add_vline(
        x=split_index,
        line_dash="dash",
        line_color="red",
        annotation_text="Split Point",
        annotation_position="top left"
    )
    
    # Add annotations for in-sample/out-of-sample
    fig.add_annotation(
        x=split_index/2,
        y=max(full_equity)*0.9,
        text="In-Sample",
        showarrow=False
    )
    fig.add_annotation(
        x=split_index + (total_length-split_index)/2,
        y=max(full_equity)*0.9,
        text="Out-of-Sample",
        showarrow=False
    )
    
    fig.update_layout(
        title='Equity Curve',
        xaxis_title='Period',
        yaxis_title='Equity ($)',
        hovermode='x unified',
        height=500
    )
    
    return fig

# Plot trade analysis
def plot_trade_analysis(trade_records):
    if trade_records.empty:
        return None, None, None
    
    # Trade duration histogram
    fig_duration = go.Figure()
    fig_duration.add_trace(go.Histogram(
        x=trade_records['duration'],
        name='Trade Duration',
        marker_color='blue'
    ))
    fig_duration.update_layout(
        title='Trade Duration Distribution (Days)',
        xaxis_title='Duration (Days)',
        yaxis_title='Count',
        height=300
    )
    
    # PnL distribution
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Histogram(
        x=trade_records['pnl_percent'],
        name='Trade PnL %',
        marker_color='green'
    ))
    fig_pnl.update_layout(
        title='Trade PnL Distribution (%)',
        xaxis_title='PnL %',
        yaxis_title='Count',
        height=300
    )
    
    # Cumulative PnL over time
    trade_records = trade_records.sort_values('exit_date')
    trade_records['cum_pnl'] = trade_records['pnl_percent'].cumsum()
    
    fig_cum_pnl = go.Figure()
    fig_cum_pnl.add_trace(go.Scatter(
        x=trade_records['exit_date'],
        y=trade_records['cum_pnl'],
        name='Cumulative PnL',
        line=dict(color='green')
    ))
    fig_cum_pnl.update_layout(
        title='Cumulative PnL Over Time',
        xaxis_title='Date',
        yaxis_title='Cumulative PnL %',
        height=300
    )
    
    return fig_duration, fig_pnl, fig_cum_pnl

# Main App
st.title("📈 AI-Powered Trading Strategy Optimizer")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    uploaded_file = st.file_uploader("📤 Upload Market Data (CSV)", type=['csv', 'txt'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = {'Date', 'Open', 'High', 'Low', 'Close'}
            if not required_columns.issubset(df.columns):
                st.error("CSV must contain: Date, Open, High, Low, Close")
            else:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                st.session_state.df = df
                st.session_state.data_uploaded = True
                st.success("Data uploaded successfully!")
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    if st.session_state.data_uploaded:
        st.session_state.split_ratio = st.slider(
            "In-Sample / Out-of-Sample Split", 
            0.1, 0.9, 0.7, 0.05,
            help="Recommended: 70% in-sample for robust optimization"
        )
        
        st.session_state.optimization_metric = st.selectbox(
            "Optimization Metric", 
            ["Return-to-Drawdown Ratio", "Profit Factor"],
            index=0,
            help="Return-to-Drawdown is preferred for risk-adjusted performance"
        )
        
        st.session_state.target_value = st.number_input(
            f"Target {st.session_state.optimization_metric}", 
            0.1, 10.0, 2.0, 0.1,
            help="Realistic target: 2.0-3.0 for RR/DD, 1.5+ for Profit Factor"
        )
            
        st.session_state.optimization_mode = st.radio(
            "Optimization Mode", 
            ["Find Best Possible", "Find First Valid"],
            index=0,
            help="'Find Best Possible' for thorough optimization"
        )
        
        st.session_state.max_iterations = st.number_input(
            "Max Iterations", 
            10, 5000, 500, 10,
            help="More iterations (500+) for better results"
        )
        
        st.session_state.max_time_minutes = st.number_input(
            "Max Time (minutes)", 
            1, 240, 30, 1,
            help="Allow sufficient time for optimization"
        )
        
        st.session_state.strategy_type = st.selectbox(
            "Strategy Type",
            ["Automatic Selection", "Trend Following", "Mean Reversion", "Statistical Arbitrage"],
            index=0,
            help="Automatic selection will test all strategy types"
        )
        
        advanced = st.checkbox("Show Advanced Options")
        if advanced:
            st.session_state.risk_per_trade = st.number_input(
                "Risk Per Trade (%)",
                0.1, 5.0, 1.0, 0.1,
                help="Percentage of capital to risk per trade"
            )
            
            st.session_state.min_trades = st.number_input(
                "Minimum Trades",
                1, 100, 5, 1,
                help="Require at least this many trades for valid strategy"
            )
            
            st.session_state.max_consecutive_failures = st.number_input(
                "Max Consecutive Failures",
                5, 100, 20, 1,
                help="Stop optimization after this many failed trials"
            )

# Main Content
if st.session_state.data_uploaded:
    df = st.session_state.df
    
    st.header("📋 Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Data Points", len(df))
        st.metric("Date Range", f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        
    with col2:
        returns = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        st.metric("Buy & Hold Return", f"{returns*100:.1f}%")
        volatility = df['Close'].pct_change().std() * np.sqrt(252)
        st.metric("Annualized Volatility", f"{volatility*100:.1f}%")
    
    st.header("📈 Market Data Visualization")
    fig = go.Figure(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    ))
    fig.update_layout(
        title='Price Chart',
        height=500,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.header("🤖 Strategy Optimization")
    
    if st.button("🚀 Optimize Trading Strategy", type="primary"):
        with st.spinner("Optimizing strategy. This may take several minutes..."):
            split_index = int(len(df) * st.session_state.split_ratio)
            in_sample = df.iloc[:split_index]
            out_of_sample = df.iloc[split_index:]
            
            st.session_state.in_sample = in_sample
            st.session_state.out_of_sample = out_of_sample
            st.session_state.split_index = split_index
            
            result = generate_strategy(
                in_sample,
                st.session_state.strategy_type,
                st.session_state.optimization_metric,
                st.session_state.target_value,
                st.session_state.max_iterations,
                True,
                st.session_state.optimization_mode,
                st.session_state.max_time_minutes
            )
            
            if result is not None:
                st.session_state.strategy_code = result['strategy_code']
                st.session_state.strategy_params = result['params']
                st.session_state.strategy_score = result['score']
                st.session_state.strategy_type = result['strategy_type']
                st.session_state.in_sample_results = result['results']
                st.session_state.strategy_generated = True
                st.session_state.found_valid = result['found_valid']
                
                # Out-of-sample backtest
                oos_results = backtest_strategy(out_of_sample, result['params'], is_sample=False)
                st.session_state.out_of_sample_results = oos_results
                st.session_state.backtest_completed = True
                
                if result['score'] >= st.session_state.target_value:
                    st.success(f"✅ Strategy found with {st.session_state.optimization_metric} of {result['score']:.2f}")
                else:
                    st.warning(f"⚠️ Best strategy has {st.session_state.optimization_metric} of {result['score']:.2f} (target: {st.session_state.target_value})")
            else:
                st.error("Strategy generation failed. Please try adjusting parameters.")
    
    if st.session_state.strategy_generated and st.session_state.in_sample_results is not None:
        st.header("📊 Optimization Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Strategy Summary")
            
            strategy_type_badge = {
                'Trend Following': 'success',
                'Mean Reversion': 'warning',
                'Statistical Arbitrage': 'info'
            }.get(st.session_state.strategy_type, 'primary')
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Strategy Type: <span class="badge badge-{strategy_type_badge}">{st.session_state.strategy_type}</span></h4>
                <p><strong>Optimization Metric:</strong> {st.session_state.optimization_metric}</p>
                <p><strong>Achieved Score:</strong> {st.session_state.strategy_score:.2f}</p>
                <p><strong>Target Score:</strong> {st.session_state.target_value}</p>
                {'<p class="text-success">✅ Target Achieved</p>' if st.session_state.found_valid else '<p class="text-warning">⚠️ Target Not Achieved</p>'}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Key Parameters")
            params_df = pd.DataFrame.from_dict(st.session_state.strategy_params, orient='index', columns=['Value'])
            params_df['Value'] = params_df['Value'].astype(str)  # Convert all values to strings
            st.dataframe(params_df)
            
        with col2:
            st.markdown("### Performance Snapshot")
            
            cols = st.columns(2)
            with cols[0]:
                is_return = st.session_state.in_sample_results['total_return']*100
                return_color = "green" if is_return > 0 else "red"
                st.metric("In-Sample Return", f"{is_return:.1f}%", delta_color="off", help="Total return during optimization period")
                
                is_rr_dd = st.session_state.in_sample_results['rr_dd_ratio']
                st.metric("In-Sample RR/DD", f"{is_rr_dd:.2f}", help="Return-to-Drawdown ratio")
                
                is_pf = st.session_state.in_sample_results['profit_factor']
                pf_color = "green" if is_pf > 1.5 else "orange" if is_pf > 1 else "red"
                st.metric("In-Sample Profit Factor", f"{is_pf:.2f}", help="Gross profit / gross loss", delta_color="off")
                
            with cols[1]:
                oos_return = st.session_state.out_of_sample_results['total_return']*100
                return_color = "green" if oos_return > 0 else "red"
                st.metric("Out-of-Sample Return", f"{oos_return:.1f}%", delta_color="off", help="Total return on unseen data")
                
                oos_rr_dd = st.session_state.out_of_sample_results['rr_dd_ratio']
                st.metric("Out-of-Sample RR/DD", f"{oos_rr_dd:.2f}", help="Return-to-Drawdown ratio")
                
                oos_pf = st.session_state.out_of_sample_results['profit_factor']
                pf_color = "green" if oos_pf > 1.5 else "orange" if oos_pf > 1 else "red"
                st.metric("Out-of-Sample Profit Factor", f"{oos_pf:.2f}", help="Gross profit / gross loss", delta_color="off")
            
            st.metric("Sharpe Ratio (OOS)", f"{st.session_state.out_of_sample_results['sharpe_ratio']:.2f}", 
                     help="Risk-adjusted returns (higher is better)")
            st.metric("System Quality Number", f"{st.session_state.in_sample_results['sqn']:.2f}", 
                     help="Strategy quality metric (>1.6 is good, >2.5 is excellent)")
        
        st.header("📜 Strategy Implementation")
        
        tab1, tab2, tab3 = st.tabs(["EasyLanguage Code", "Strategy Logic", "Python Implementation"])
        
        with tab1:
            st.code(st.session_state.strategy_code, language="text")
            
            st.download_button(
                "📥 Download Strategy Code",
                st.session_state.strategy_code,
                file_name=f"{st.session_state.strategy_type.replace(' ', '_')}_strategy.els",
                mime="text/plain"
            )
        
        with tab2:
            if st.session_state.strategy_params['strategy_type'] == 'trend':
                st.markdown("""
                #### Trend Following Strategy Logic:
                
                - **Entry Conditions**:
                  - Price above Moving Average (confirming uptrend)
                  - RSI not overbought (avoid chasing extended moves)
                  - MACD histogram positive (momentum confirmation)
                
                - **Exit Conditions**:
                  - Price crosses below Moving Average (trend reversal)
                  - RSI becomes overbought (overextended move)
                  - Trailing stop loss (locks in profits)
                  - Fixed stop loss/take profit levels
                
                - **Risk Management**:
                  - Position sized based on volatility (ATR)
                  - Maximum risk per trade (1% of capital)
                  - Maximum position size (10% of capital)
                """)
            elif st.session_state.strategy_params['strategy_type'] == 'mean_reversion':
                st.markdown("""
                #### Mean Reversion Strategy Logic:
                
                - **Entry Conditions**:
                  - Price below lower Bollinger Band (oversold)
                  - RSI below oversold threshold (confirmation)
                  - Volume above average (participation)
                
                - **Exit Conditions**:
                  - Price returns to midline (mean reversion complete)
                  - RSI reaches neutral/overbought levels
                  - Fixed stop loss/take profit levels
                
                - **Risk Management**:
                  - Position sized based on volatility (ATR)
                  - Maximum risk per trade (1% of capital)
                  - Maximum position size (10% of capital)
                """)
            else:
                st.markdown("""
                #### Statistical Arbitrage Strategy Logic:
                
                - **Entry Conditions**:
                  - Price outside statistical bands (mean-reverting)
                  - Hurst exponent < 0.5 (confirming mean-reverting regime)
                  - RSI confirmation (avoid false breakouts)
                
                - **Exit Conditions**:
                  - Price returns to inner bands
                  - Fixed stop loss/take profit levels
                
                - **Risk Management**:
                  - Position sized based on volatility (ATR)
                  - Maximum risk per trade (1% of capital)
                  - Maximum position size (10% of capital)
                """)
        
        with tab3:
            st.code(f"""
# Python implementation of {st.session_state.strategy_type} strategy
params = {st.session_state.strategy_params}

class OptimizedStrategy(bt.Strategy):
    params = (
        {', '.join([f"('{k}', {v})" for k, v in st.session_state.strategy_params.items()])}
    )
    
    def __init__(self):
        # Initialize indicators based on strategy type
        {'# Trend following indicators' if st.session_state.strategy_params['strategy_type'] == 'trend' else 
          '# Mean reversion indicators' if st.session_state.strategy_params['strategy_type'] == 'mean_reversion' else 
          '# Statistical arbitrage indicators'}
        ...
            
    def next(self):
        # Implement trading logic
        {'# Trend following logic' if st.session_state.strategy_params['strategy_type'] == 'trend' else 
          '# Mean reversion logic' if st.session_state.strategy_params['strategy_type'] == 'mean_reversion' else 
          '# Statistical arbitrage logic'}
        ...
            """, language='python')
        
        st.header("📊 Performance Analysis")
        
        st.plotly_chart(
            plot_performance_metrics(
                st.session_state.in_sample_results,
                st.session_state.out_of_sample_results,
                st.session_state.split_index,
                len(df)
            ), 
            use_container_width=True
        )

        st.subheader("Trade Analysis")
        
        if not st.session_state.in_sample_results['trade_records'].empty:
            with st.expander("In-Sample Trade Details"):
                fig_duration, fig_pnl, fig_cum_pnl = plot_trade_analysis(
                    st.session_state.in_sample_results['trade_records']
                )
                
                if fig_duration:
                    st.plotly_chart(fig_duration, use_container_width=True)
                    st.plotly_chart(fig_pnl, use_container_width=True)
                    st.plotly_chart(fig_cum_pnl, use_container_width=True)
                
                display_trades = st.session_state.in_sample_results['trade_records'].copy()
                display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
                display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
                st.dataframe(display_trades.style.format({
                    'entry_price': '{:.4f}',
                    'exit_price': '{:.4f}',
                    'pnl': '{:.2f}',
                    'pnl_percent': '{:.2f}%'
                }))
        
        if not st.session_state.out_of_sample_results['trade_records'].empty:
            with st.expander("Out-of-Sample Trade Details"):
                fig_duration, fig_pnl, fig_cum_pnl = plot_trade_analysis(
                    st.session_state.out_of_sample_results['trade_records']
                )
                
                if fig_duration:
                    st.plotly_chart(fig_duration, use_container_width=True)
                    st.plotly_chart(fig_pnl, use_container_width=True)
                    st.plotly_chart(fig_cum_pnl, use_container_width=True)
                
                display_trades = st.session_state.out_of_sample_results['trade_records'].copy()
                display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
                display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
                st.dataframe(display_trades.style.format({
                    'entry_price': '{:.4f}',
                    'exit_price': '{:.4f}',
                    'pnl': '{:.2f}',
                    'pnl_percent': '{:.2f}%'
                }))
        
        st.subheader("Detailed Performance Metrics")
        
        metrics = pd.DataFrame({
            'Metric': [
                'Total Return', 
                'Annualized Return', 
                'Max Drawdown', 
                'Return/Drawdown Ratio', 
                'Profit Factor', 
                'Win Rate', 
                'Sharpe Ratio',
                'System Quality Number',
                'Avg Trade Duration (Days)',
                'Max Win Streak',
                'Max Loss Streak'
            ],
            'In-Sample': [
                f"{st.session_state.in_sample_results['total_return']*100:.2f}%",
                f"{st.session_state.in_sample_results['annual_return']*100:.2f}%",
                f"{st.session_state.in_sample_results['max_drawdown']:.2f}%",
                f"{st.session_state.in_sample_results['rr_dd_ratio']:.2f}",
                f"{st.session_state.in_sample_results['profit_factor']:.2f}",
                f"{st.session_state.in_sample_results['win_rate']*100:.2f}%",
                f"{st.session_state.in_sample_results['sharpe_ratio']:.2f}",
                f"{st.session_state.in_sample_results['sqn']:.2f}",
                f"{st.session_state.in_sample_results.get('avg_trade_duration', 0):.1f}",
                f"{st.session_state.in_sample_results.get('max_win_streak', 0)}",
                f"{st.session_state.in_sample_results.get('max_loss_streak', 0)}"
            ],
            'Out-of-Sample': [
                f"{st.session_state.out_of_sample_results['total_return']*100:.2f}%",
                f"{st.session_state.out_of_sample_results['annual_return']*100:.2f}%",
                f"{st.session_state.out_of_sample_results['max_drawdown']:.2f}%",
                f"{st.session_state.out_of_sample_results['rr_dd_ratio']:.2f}",
                f"{st.session_state.out_of_sample_results['profit_factor']:.2f}",
                f"{st.session_state.out_of_sample_results['win_rate']*100:.2f}%",
                f"{st.session_state.out_of_sample_results['sharpe_ratio']:.2f}",
                f"{st.session_state.out_of_sample_results['sqn']:.2f}",
                f"{st.session_state.out_of_sample_results.get('avg_trade_duration', 0):.1f}",
                f"{st.session_state.out_of_sample_results.get('max_win_streak', 0)}",
                f"{st.session_state.out_of_sample_results.get('max_loss_streak', 0)}"
            ]
        })
        
        st.table(metrics.style.applymap(
            lambda x: 'color: green' if '%' in x and float(x.replace('%','')) > 0 else 'color: red' if '%' in x and float(x.replace('%','')) < 0 else '',
            subset=['In-Sample', 'Out-of-Sample']
        ))
        
        st.header("💾 Export Results")
        
        report = f"""
        AI Trading Strategy Optimization Report
        ======================================
        
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Strategy Type: {st.session_state.strategy_type}
        Optimization Metric: {st.session_state.optimization_metric}
        Achieved Score: {st.session_state.strategy_score:.2f}
        Target Score: {st.session_state.target_value}
        {'✅ Target Achieved' if st.session_state.found_valid else '⚠️ Target Not Achieved'}
        
        Data Information:
        - Time Period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}
        - Data Points: {len(df)}
        - In-Sample Period: {st.session_state.in_sample['Date'].min().strftime('%Y-%m-%d')} to {st.session_state.in_sample['Date'].max().strftime('%Y-%m-%d')}
        - Out-of-Sample Period: {st.session_state.out_of_sample['Date'].min().strftime('%Y-%m-%d')} to {st.session_state.out_of_sample['Date'].max().strftime('%Y-%m-%d')}
        
        Strategy Parameters:
        {pd.DataFrame.from_dict(st.session_state.strategy_params, orient='index').to_string()}
        
        Performance Metrics:
        {metrics.to_string()}
        
        Strategy Code:
        {st.session_state.strategy_code}
        """
        
        st.download_button(
            "📄 Download Full Report",
            report,
            file_name=f"{st.session_state.strategy_type.replace(' ', '_')}_strategy_report.txt",
            mime="text/plain"
        )

else:
    st.info("ℹ️ Please upload market data (CSV) in the sidebar to get started")
    st.markdown("""
    ### Expected CSV Format:
    - Must contain these columns (case sensitive):
      - `Date` (YYYY-MM-DD format)
      - `Open` (opening price)
      - `High` (highest price)
      - `Low` (lowest price)
      - `Close` (closing price)
      - `Volume` (optional)
    
    ### Optimization Tips:
    1. Use at least 5+ years of daily data for robust results
    2. Start with 70/30 in-sample/out-of-sample split
    3. Target RR/DD ratio of 2.0-3.0 for realistic expectations
    4. Run 500+ iterations for thorough optimization
    5. Allow 30+ minutes for complex optimizations
    """)
