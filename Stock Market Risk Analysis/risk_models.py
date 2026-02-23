import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

# ==============================================================================
# 1. DATA INTERFACE (Dependency Inversion Principle)
# ==============================================================================
class IMarketDataFetcher(ABC):
    """
    Abstract interface for fetching raw market and financial data.
    Ensures that high-level modules do not depend on low-level API details.
    """
    @abstractmethod
    def fetch_financials(self, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def fetch_market_data(self, ticker: str) -> Dict[str, float]:
        pass

# ==============================================================================
# 2. DATA IMPLEMENTATION (Single Responsibility Principle)
# ==============================================================================
class YahooFinanceFetcher(IMarketDataFetcher):
    """
    Concrete implementation of IMarketDataFetcher using the yfinance library.
    Responsible exclusively for communicating with the external API.
    """
    def fetch_financials(self, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        stock = yf.Ticker(ticker)
        return stock.balance_sheet, stock.income_stmt

    def fetch_market_data(self, ticker: str) -> Dict[str, float]:
        stock = yf.Ticker(ticker)
        data = {'market_cap': 0.0, 'current_price': 0.0, 'equity_volatility': 0.0}
        
        try:
            self._populate_price_and_cap(stock, data)
            self._populate_volatility(stock, data)
        except Exception as e:
            print(f"Error fetching market data for {ticker}: {e}")
            
        return data

    def _populate_price_and_cap(self, stock: yf.Ticker, data: Dict[str, float]) -> None:
        """Helper method to extract current price and market capitalization."""
        shares = stock.info.get('sharesOutstanding', 0)
        history = stock.history(period="1y")
        
        if not history.empty:
            price = history['Close'].iloc[-1]
            data['current_price'] = price
            data['market_cap'] = shares * price

    def _populate_volatility(self, stock: yf.Ticker, data: Dict[str, float]) -> None:
        """Helper method to calculate annualized equity volatility."""
        history = stock.history(period="1y")
        
        if not history.empty:
            log_returns = np.log(history['Close'] / history['Close'].shift(1))
            data['equity_volatility'] = log_returns.std() * np.sqrt(252)

# ==============================================================================
# 3. FINANCIAL PROCESSOR (Single Responsibility Principle)
# ==============================================================================
class FinancialProcessor:
    """
    Responsible for extracting specific accounting line items from raw dataframes.
    Handles column name variations and basic data cleaning safely.
    """
    def __init__(self, balance_sheet: pd.DataFrame, income_stmt: pd.DataFrame):
        self.bs = balance_sheet
        self.is_ = income_stmt

    def get_total_assets(self) -> float:
        return self._safe_get(self.bs, 'Total Assets')

    def get_total_liabilities(self) -> float:
        return self._safe_get(self.bs, 'Total Liabilities Net Minority Interest')
        
    def get_total_debt(self) -> float:
        return self._safe_get(self.bs, 'Total Debt')

    def get_current_assets(self) -> float:
        return self._safe_get(self.bs, 'Current Assets')

    def get_current_liabilities(self) -> float:
        return self._safe_get(self.bs, 'Current Liabilities')

    def get_retained_earnings(self) -> float:
        return self._safe_get(self.bs, 'Retained Earnings')

    def get_stockholders_equity(self) -> float:
        return self._safe_get(self.bs, 'Stockholders Equity')

    def get_ebit(self) -> float:
        val = self._safe_get(self.is_, 'EBIT')
        return val if val != 0 else self._safe_get(self.is_, 'Operating Income')

    def get_total_revenue(self) -> float:
        return self._safe_get(self.is_, 'Total Revenue')

    def _safe_get(self, df: pd.DataFrame, key: str) -> float:
        """Safely retrieves a float value from a dataframe, handling missing keys."""
        try:
            if key in df.index: 
                return float(df.loc[key].iloc[0])
            for idx in df.index:
                if key.lower() in str(idx).lower(): 
                    return float(df.loc[idx].iloc[0])
            return 0.0
        except Exception:
            return 0.0

# ==============================================================================
# 4. STRATEGY INTERFACE FOR MODELS (Open/Closed Principle)
# ==============================================================================
class IRiskModelStrategy(ABC):
    """
    Strategy Interface. Defines the contract for all risk calculation models.
    Allows adding new models without modifying existing orchestrator code.
    """
    @abstractmethod
    def calculate(self, processor: FinancialProcessor, market_data: Dict[str, float], **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def interpret(self, score: float) -> str:
        pass

# ==============================================================================
# 5. ALTMAN STRATEGIES 
# ==============================================================================
class OriginalAltmanStrategy(IRiskModelStrategy):
    """
    Implements the Original 1968 Altman Z-Score for manufacturing firms.
    """
    def calculate(self, processor: FinancialProcessor, market_data: Dict[str, float], **kwargs) -> Dict[str, Any]:
        ta = processor.get_total_assets()
        if ta == 0: 
            return {}

        ratios = self._compute_ratios(processor, market_data, ta)
        z_score = self._compute_z_score(ratios)
        
        return {
            "score": round(z_score, 4),
            "metric_name": "Z-Score",
            "components": ratios,
            "model_name": "Original Altman Z-Score (1968)"
        }

    def _compute_ratios(self, processor: FinancialProcessor, market_data: Dict[str, float], ta: float) -> Dict[str, float]:
        """Calculates the 5 individual components of the original Z-score."""
        wc = processor.get_current_assets() - processor.get_current_liabilities()
        tl = processor.get_total_liabilities()
        
        return {
            "X1 (Liquidity)": wc / ta,
            "X2 (Profitability)": processor.get_retained_earnings() / ta,
            "X3 (Operating)": processor.get_ebit() / ta,
            "X4 (Market)": market_data['market_cap'] / tl if tl != 0 else 0,
            "X5 (Activity)": processor.get_total_revenue() / ta
        }

    def _compute_z_score(self, ratios: Dict[str, float]) -> float:
        """Applies the coefficients to the calculated ratios."""
        return (1.2 * ratios["X1 (Liquidity)"] + 
                1.4 * ratios["X2 (Profitability)"] + 
                3.3 * ratios["X3 (Operating)"] + 
                0.6 * ratios["X4 (Market)"] + 
                1.0 * ratios["X5 (Activity)"])

    def interpret(self, z: float) -> str:
        if z > 2.99: return "Safe Zone"
        elif z < 1.81: return "Distress Zone"
        else: return "Grey Zone"


class EmergingMarketsAltmanStrategy(IRiskModelStrategy):
    """
    Implements the Altman Z'' Score tailored for Non-Manufacturing / Emerging Markets.
    """
    def calculate(self, processor: FinancialProcessor, market_data: Dict[str, float], **kwargs) -> Dict[str, Any]:
        ta = processor.get_total_assets()
        if ta == 0: 
            return {}

        ratios = self._compute_ratios(processor, ta)
        z_score = self._compute_z_score(ratios)
        
        return {
            "score": round(z_score, 4),
            "metric_name": "Z-Score",
            "components": ratios,
            "model_name": "Altman Z'' Score (Non-Manuf / Emerging)"
        }

    def _compute_ratios(self, processor: FinancialProcessor, ta: float) -> Dict[str, float]:
        """Calculates the 4 individual components of the Z''-score."""
        wc = processor.get_current_assets() - processor.get_current_liabilities()
        tl = processor.get_total_liabilities()
        
        return {
            "X1 (Liquidity)": wc / ta,
            "X2 (Profitability)": processor.get_retained_earnings() / ta,
            "X3 (Operating)": processor.get_ebit() / ta,
            "X4 (Book Value)": processor.get_stockholders_equity() / tl if tl != 0 else 0
        }

    def _compute_z_score(self, ratios: Dict[str, float]) -> float:
        """Applies the specific coefficients for emerging markets."""
        return (6.56 * ratios["X1 (Liquidity)"] + 
                3.26 * ratios["X2 (Profitability)"] + 
                6.72 * ratios["X3 (Operating)"] + 
                1.05 * ratios["X4 (Book Value)"])

    def interpret(self, z: float) -> str:
        if z > 2.6: return "Safe Zone"
        elif z < 1.1: return "Distress Zone"
        else: return "Grey Zone"

# ==============================================================================
# 6. MERTON STRATEGY
# ==============================================================================
class MertonModelStrategy(IRiskModelStrategy):
    """
    Merton Distance to Default Model.
    Treats equity as a call option on the company's assets using Black-Scholes.
    Utilizes numerical optimization to solve for unobservable asset variables.
    """
    def calculate(self, processor: FinancialProcessor, market_data: Dict[str, float], **kwargs) -> Dict[str, Any]:
        E = market_data.get('market_cap', 0)
        sigma_E = market_data.get('equity_volatility', 0)
        D = processor.get_total_debt()
        r = kwargs.get('risk_free_rate', 0.035)
        T = kwargs.get('maturity_years', 1.0)

        if not self._validate_inputs(D, E, sigma_E): 
            return {}

        V, sigma_V, market_leverage = self._solve_merton_system(E, D, sigma_E, r, T)
        DD = self._compute_distance_to_default(V, D, sigma_V, r, T)
        PD = self._compute_probability_of_default(DD)
        
        return self._build_result_dict(PD, DD, sigma_V, market_leverage, V, sigma_E, r)

    def _validate_inputs(self, D: float, E: float, sigma_E: float) -> bool:
        """Ensures non-zero values for critical unobservable proxy inputs."""
        return D != 0 and E != 0 and sigma_E != 0

    @staticmethod
    def _merton_equations(x: np.ndarray, E: float, sigma_E: float, D: float, r: float, T: float) -> np.ndarray:
        """
        Defines the non-linear system of equations based on Black-Scholes pricing.
        Equation 1: Equity value as a call option.
        Equation 2: Volatility relationship based on Ito's Lemma.
        """
        V, sigma_V = x
        
        if V <= 0 or sigma_V <= 0:
            return np.array([1e10, 1e10])
        
        d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
        d2 = d1 - sigma_V * np.sqrt(T)
        
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        
        eq1 = V * N_d1 - D * np.exp(-r * T) * N_d2 - E
        eq2 = (V / E) * N_d1 * sigma_V - sigma_E
        
        return np.array([eq1, eq2])

    def _solve_merton_system(self, E: float, D: float, sigma_E: float, r: float, T: float) -> Tuple[float, float, float]:
        """
        Solves for Asset Value Proxy (V) and Asset Volatility (sigma_V) using fsolve.
        """
        V_init = E + D
        sigma_V_init = sigma_E * (E / V_init)
        
        solution = fsolve(self._merton_equations, x0=[V_init, sigma_V_init], args=(E, sigma_E, D, r, T))
        V, sigma_V = solution
        
        market_leverage = D / V if V != 0 else 0
        return V, sigma_V, market_leverage

    def _compute_distance_to_default(self, V: float, D: float, sigma_V: float, r: float, T: float) -> float:
        """Calculates the Distance to Default (DD) using the structural model equation."""
        numerator = np.log(V / D) + (r - 0.5 * sigma_V**2) * T
        denominator = sigma_V * np.sqrt(T)
        return numerator / denominator

    def _compute_probability_of_default(self, DD: float) -> float:
        """Transforms Distance to Default into Probability of Default (PD)."""
        return 1.0 - norm.cdf(DD)

    def _build_result_dict(self, PD: float, DD: float, sigma_V: float, leverage: float, 
                           V: float, sigma_E: float, r: float) -> Dict[str, Any]:
        """Constructs the standardized output dictionary."""
        return {
            "score": PD,  
            "metric_name": "Probability of Default",
            "components": {
                "Distance to Default (DD)": DD,
                "Asset Volatility (σ_V)": sigma_V,
                "Market Leverage": leverage,
                "Asset Value Proxy (V)": V,
                "Equity Volatility (σ_E)": sigma_E,
                "Risk-Free Rate (r)": r
            },
            "model_name": "Merton Structural Model"
        }

    def interpret(self, pd: float) -> str:
        if pd < 0.02: return "Safe Zone (Low Default Probability)"
        elif pd > 0.05: return "Distress Zone (High Default Probability)"
        else: return "Grey Zone (Caution)"

# ==============================================================================
# 7. MAIN SERVICE ORCHESTRATOR
# ==============================================================================
class CreditAnalystService:
    """
    Coordinates data fetching, financial processing, and strategy execution.
    Acts as the main API for external UI components to request analysis.
    """
    def __init__(self, fetcher: IMarketDataFetcher):
        self.fetcher = fetcher

    def analyze_company(self, ticker: str, strategy: IRiskModelStrategy, **kwargs) -> Dict[str, Any]:
        bs, is_ = self.fetcher.fetch_financials(ticker)
        market_data = self.fetcher.fetch_market_data(ticker)
        processor = FinancialProcessor(bs, is_)
        
        result = strategy.calculate(processor, market_data, **kwargs)
        
        if not result:
            return {"error": "Insufficient data to run the model."}

        return self._format_response(ticker, result, strategy)

    def _format_response(self, ticker: str, result: Dict[str, Any], strategy: IRiskModelStrategy) -> Dict[str, Any]:
        """Combines the raw calculation results with high-level contextual data."""
        zone = strategy.interpret(result['score'])
        return {
            "Ticker": ticker,
            "Model": result['model_name'],
            "Metric Name": result['metric_name'],
            "Score": result['score'],
            "Zone": zone,
            "Details": result['components']
        }