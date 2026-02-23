import streamlit as st
import pandas as pd
import plotly.express as px
from risk_models import (
    YahooFinanceFetcher, 
    CreditAnalystService, 
    OriginalAltmanStrategy, 
    EmergingMarketsAltmanStrategy,
    MertonModelStrategy
)

# ==============================================================================
# 1. UI SETUP AND CONFIGURATION
# ==============================================================================
class UIConfigurator:
    """Handles the main page configuration and static headers."""
    @staticmethod
    def setup_page():
        st.set_page_config(page_title="Multi-Model Credit Risk Dashboard", layout="wide")
        st.title("📊 Comprehensive Corporate Credit Risk Analyzer")
        st.markdown("Evaluating 3 companies simultaneously across Altman and Merton models.")

# ==============================================================================
# 2. INPUT MANAGEMENT
# ==============================================================================
class SidebarInputManager:
    """Responsible for rendering the sidebar and collecting user inputs."""
    def render_and_get_inputs(self) -> tuple:
        st.sidebar.header("🏢 Select Companies")
        t1 = st.sidebar.text_input("Company 1 Ticker:", value="AAPL").upper()
        t2 = st.sidebar.text_input("Company 2 Ticker:", value="FCX").upper()
        t3 = st.sidebar.text_input("Company 3 Ticker:", value="NFLX").upper()
        
        tickers = [t.strip() for t in [t1, t2, t3] if t.strip()]

        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 📈 Merton Parameters")
        rf_input = st.sidebar.number_input(
            "Risk-Free Interest Rate (%)", 
            min_value=0.0, 
            max_value=20.0, 
            value=3.5, 
            step=0.1,
            help="Default is 3.5%. This is the 'r' parameter in the Black-Scholes formula."
        )
        risk_free_rate = rf_input / 100.0

        run_btn = st.sidebar.button("Analyze Risk", type="primary", use_container_width=True)
        return tickers, risk_free_rate, run_btn

# ==============================================================================
# 3. ANALYSIS CONTROLLER (Orchestrator)
# ==============================================================================
class RiskAnalysisController:
    """Coordinates data fetching and execution across all selected models."""
    def __init__(self):
        self.fetcher = YahooFinanceFetcher()
        self.service = CreditAnalystService(self.fetcher)
        self.strategies = {
            "Altman Z-Score Original (1968)": OriginalAltmanStrategy(),
            "Altman Z'' Score (Emerging Markets)": EmergingMarketsAltmanStrategy(),
            "Merton Model (Probability of Default)": MertonModelStrategy()
        }

    def run_all_models(self, tickers: list, risk_free_rate: float) -> tuple:
        model_results = {}
        final_evaluations = {ticker: {} for ticker in tickers}

        for model_name, strategy in self.strategies.items():
            results_list = []
            
            for ticker in tickers:
                data = self.service.analyze_company(ticker, strategy, risk_free_rate=risk_free_rate)
                
                if "error" not in data:
                    final_evaluations[ticker][model_name] = data["Zone"]
                    results_list.append(self._build_result_dict(data, model_name))
                else:
                    st.error(f"Error analyzing {ticker}: {data['error']}")
            
            model_results[model_name] = results_list

        return model_results, final_evaluations

    def _build_result_dict(self, data: dict, model_name: str) -> dict:
        """Parses model output to a standard dictionary format."""
        result_dict = {
            "Ticker": data["Ticker"],
            "Score": data["Score"],
            "Zone": data["Zone"],
            "Metric": data["Metric Name"]
        }
        
        if "Merton" in model_name:
            details = data["Details"]
            result_dict["Distance to Default"] = details["Distance to Default (DD)"]
            result_dict["Asset Volatility"] = details["Asset Volatility (σ_V)"]
            result_dict["Market Leverage"] = details["Market Leverage"]
            
        return result_dict

# ==============================================================================
# 4. CHART GENERATOR
# ==============================================================================
class ChartGenerator:
    """Handles the creation of Plotly charts."""
    COLOR_MAP = {
        "Safe Zone": "#00CC96",
        "Safe Zone (Low Default Probability)": "#00CC96",
        "Grey Zone": "#FFA15A",
        "Grey Zone (Caution)": "#FFA15A",
        "Distress Zone": "#EF553B",
        "Distress Zone (High Default Probability)": "#EF553B"
    }

    @staticmethod
    def create_comparative_bar_chart(df: pd.DataFrame, model_name: str):
        fig = px.bar(
            df, 
            x="Ticker", 
            y="Score", 
            color="Zone",
            text="Score",
            title=f"Comparison: {df['Metric'].iloc[0]}",
            color_discrete_map=ChartGenerator.COLOR_MAP
        )
        
        if "Merton" in model_name:
            fig.update_traces(texttemplate='%{text:.4%}', textposition='outside')
            fig.update_layout(yaxis_title="Probability of Default", yaxis_tickformat='.2%')
        else:
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.update_layout(yaxis_title="Z-Score")
            
        return fig

# ==============================================================================
# 5. RESULTS RENDERER
# ==============================================================================
class ModelResultsRenderer:
    """Responsible for displaying the dataframes and charts for each model."""
    def render(self, model_results: dict):
        for model_name, results_list in model_results.items():
            st.header(f"📌 {model_name}")
            
            if not results_list:
                st.divider()
                continue
                
            df = pd.DataFrame(results_list)
            col1, col2 = st.columns([1.2, 1])
            
            with col1:
                st.markdown("**Numerical Results**")
                display_df = self._format_dataframe(df, model_name)
                st.dataframe(display_df, use_container_width=True)
                
            with col2:
                st.markdown("**Comparative Chart**")
                fig = ChartGenerator.create_comparative_bar_chart(df, model_name)
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()

    def _format_dataframe(self, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Formats the numbers for UI presentation depending on the model."""
        display_df = df.copy()
        
        if "Merton" in model_name:
            display_df["Probability of Default"] = display_df["Score"].apply(lambda x: f"{x * 100:.4f}%")
            display_df["Distance to Default"] = display_df["Distance to Default"].apply(lambda x: f"{x:.4f}")
            display_df["Asset Volatility"] = display_df["Asset Volatility"].apply(lambda x: f"{x * 100:.4f}%")
            display_df["Market Leverage"] = display_df["Market Leverage"].apply(lambda x: f"{x * 100:.4f}%")
            return display_df[["Ticker", "Probability of Default", "Distance to Default", "Asset Volatility", "Market Leverage", "Zone"]]
        else:
            display_df["Score"] = display_df["Score"].apply(lambda x: f"{x:.4f}")
            return display_df[["Ticker", "Score", "Zone"]]

# ==============================================================================
# 6. FINAL DECISION RENDERER
# ==============================================================================
class FinalDecisionRenderer:
    """Handles the logic and rendering of the final credit approval decision."""
    def render(self, tickers: list, final_evaluations: dict):
        st.header("⚖️ Final Credit Decision")
        st.markdown("📝 **Rule:** The loan is approved ONLY if at least one Altman model is in the 'Safe Zone' **AND** the Merton model is in the 'Safe Zone'.")
        
        cols = st.columns(3)
        for idx, ticker in enumerate(tickers):
            with cols[idx]:
                st.subheader(f"🏢 {ticker}")
                
                if ticker in final_evaluations and len(final_evaluations[ticker]) == 3:
                    self._evaluate_and_display(ticker, final_evaluations[ticker])
                else:
                    st.warning("⚠️ Insufficient data to make a decision.")

    def _evaluate_and_display(self, ticker: str, evals: dict):
        """Processes the rules to approve or deny the credit."""
        altman_1_zone = evals.get("Altman Z-Score Original (1968)", "")
        altman_2_zone = evals.get("Altman Z'' Score (Emerging Markets)", "")
        merton_zone = evals.get("Merton Model (Probability of Default)", "")
        
        altman_approved = "Safe" in altman_1_zone or "Safe" in altman_2_zone
        merton_approved = "Safe" in merton_zone
        
        st.markdown(f"- **Altman (Orig):** {altman_1_zone}")
        st.markdown(f"- **Altman (Z''):** {altman_2_zone}")
        st.markdown(f"- **Merton (PD):** {merton_zone}")
        st.markdown("---")
        
        if altman_approved and merton_approved:
            st.success("✅ **APPROVED**\n\nThe company meets the safety criteria for both the structural (Merton) and accounting (Altman) models.")
        else:
            st.error("❌ **DENIED**\n\nThe company does not meet the combined safety criteria required for credit approval.")

# ==============================================================================
# 7. MAIN APPLICATION (Entry Point)
# ==============================================================================
class DashboardApp:
    """Main application orchestrator combining all specific UI components."""
    def run(self):
        UIConfigurator.setup_page()
        
        input_manager = SidebarInputManager()
        tickers, risk_free_rate, run_btn = input_manager.render_and_get_inputs()

        if run_btn:
            if len(tickers) != 3:
                st.sidebar.error("Please enter exactly 3 valid tickers.")
                return
                
            with st.spinner("Fetching data and running all models..."):
                # 1. Orchestrate Analysis
                controller = RiskAnalysisController()
                model_results, final_evaluations = controller.run_all_models(tickers, risk_free_rate)
                
                # 2. Render Results Blocks
                results_renderer = ModelResultsRenderer()
                results_renderer.render(model_results)
                
                # 3. Render Final Decision Block
                decision_renderer = FinalDecisionRenderer()
                decision_renderer.render(tickers, final_evaluations)

if __name__ == "__main__":
    app = DashboardApp()
    app.run()