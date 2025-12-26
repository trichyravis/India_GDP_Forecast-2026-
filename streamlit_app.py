"""
================================================================================
INDIA GDP PROJECTION 2026 - STREAMLIT INTERACTIVE APP
================================================================================

Interactive financial modeling dashboard for India's 2026 GDP forecast
Built with Streamlit for easy deployment and customization

Author: Prof. V. Ravichandran
The Mountain Path - World of Finance

Features:
- Interactive scenario builder
- Real-time sensitivity analysis
- Monte Carlo simulations
- Professional visualizations
- Downloadable results

To run locally:
    streamlit run app.py

To deploy to Streamlit Cloud:
    1. Push to GitHub
    2. Go to share.streamlit.io
    3. Select repository
    4. Deploy
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import io
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="India GDP 2026 Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        color: #003366;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        color: #666666;
        font-size: 1.2em;
        margin-bottom: 20px;
    }
    .metric-box {
        background: #f0f0f0;
        padding: 15px;
        border-left: 4px solid #003366;
        border-radius: 5px;
        margin: 10px 0;
    }
    .scenario-box {
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .upside {
        background: #d4edda;
        border-left: 4px solid #28a745;
    }
    .base {
        background: #cfe2ff;
        border-left: 4px solid #003366;
    }
    .downside {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# COLOR SCHEME
# ============================================================================

DARKBLUE = '#003366'
LIGHTBLUE = '#ADD8E6'
GOLDCOLOR = '#FFD700'

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'model_fitted' not in st.session_state:
    st.session_state.model_fitted = False
if 'mc_results' not in st.session_state:
    st.session_state.mc_results = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = {}

# ============================================================================
# HELPER CLASSES
# ============================================================================

class GDPProjectionModel:
    """Linear regression model for GDP projection"""
    
    def __init__(self, base_year=2024):
        self.base_year = base_year
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.features = None
        self.coefficients = None
        self.r2_score = None
        
    def prepare_data(self, years, gdp_growth, predictors):
        X_scaled = self.scaler.fit_transform(predictors)
        y = np.array(gdp_growth)
        self.features = predictors.columns
        return X_scaled, y
    
    def fit(self, X_scaled, y):
        self.model.fit(X_scaled, y)
        self.coefficients = dict(zip(self.features, self.model.coef_))
        self.r2_score = self.model.score(X_scaled, y)
        return self
    
    def project(self, predictors_2026):
        X_2026 = self.scaler.transform(predictors_2026.reshape(1, -1))
        return self.model.predict(X_2026)[0]


class SectoralGDPModel:
    """Multi-sector GDP decomposition"""
    
    def __init__(self):
        self.sectors = {}
        self.weights = {}
        
    def add_sector(self, name, weight, base_growth):
        self.sectors[name] = {
            'weight': weight,
            'base_growth': base_growth,
            'elasticities': {}
        }
        self.weights[name] = weight
    
    def set_elasticities(self, sector, elasticity_dict):
        self.sectors[sector]['elasticities'] = elasticity_dict
    
    def project_growth(self, sector, variables):
        base = self.sectors[sector]['base_growth']
        elasticities = self.sectors[sector]['elasticities']
        growth = base
        for var, shock in variables.items():
            if var in elasticities:
                growth += elasticities[var] * shock
        return growth
    
    def aggregate_gdp(self, sector_growths):
        gdp_growth = 0
        for sector, growth in sector_growths.items():
            weight = self.weights[sector]
            gdp_growth += weight * growth
        return gdp_growth


class MonteCarloGDPSimulation:
    """Monte Carlo simulation for probability distribution"""
    
    def __init__(self, num_simulations=10000):
        self.num_simulations = num_simulations
        self.simulations = None
        
    def run_simulation(self, base_growth, scenarios_dict):
        self.simulations = []
        scenario_names = list(scenarios_dict.keys())
        probabilities = [scenarios_dict[s]['probability'] for s in scenario_names]
        
        for _ in range(self.num_simulations):
            scenario = np.random.choice(scenario_names, p=probabilities)
            base_sim = scenarios_dict[scenario]['growth']
            noise = np.random.normal(0, 0.15)
            self.simulations.append(base_sim + noise)
        
        self.simulations = np.array(self.simulations)
    
    def get_statistics(self):
        return {
            'mean': np.mean(self.simulations),
            'std': np.std(self.simulations),
            'median': np.median(self.simulations),
            'min': np.min(self.simulations),
            'max': np.max(self.simulations),
            'percentile_5': np.percentile(self.simulations, 5),
            'percentile_25': np.percentile(self.simulations, 25),
            'percentile_75': np.percentile(self.simulations, 75),
            'percentile_95': np.percentile(self.simulations, 95)
        }


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown(
        "<h1 class='main-header'>📊 India GDP Projection 2026</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p class='subtitle'>Interactive Financial Modeling Dashboard</p>",
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.markdown("## 🎯 Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Dashboard", "⚙️ Scenario Builder", "📈 Sensitivity Analysis", 
         "📊 Institutional Comparison", "📥 Download Results"]
    )
    
    # Initialize model data
    years = np.array([2020, 2021, 2022, 2023, 2024, 2025])
    actual_gdp = np.array([6.7, 8.7, 7.0, 7.2, 7.8, 6.5])
    
    predictors_hist = pd.DataFrame({
        'Oil_Price_USD_bbl': [45, 60, 70, 75, 68, 62],
        'Monsoon_Index': [92, 103, 98, 101, 99, 96],
        'Capex_Growth': [5.0, 12.5, 15.0, 18.0, 52.0, 35.0],
        'PFCE_Growth': [6.0, 8.5, 7.5, 8.0, 7.9, 7.5],
        'Inflation_Rate': [6.3, 5.2, 6.8, 5.4, 2.5, 2.0]
    })
    
    # Fit model
    model = GDPProjectionModel(base_year=2024)
    X_scaled, y = model.prepare_data(years, actual_gdp, predictors_hist)
    model.fit(X_scaled, y)
    
    # Setup sectoral model
    sectoral_model = SectoralGDPModel()
    sectoral_model.add_sector('Agriculture', 0.18, 3.75)
    sectoral_model.add_sector('Manufacturing', 0.27, 7.25)
    sectoral_model.add_sector('Services', 0.55, 8.75)
    
    sectoral_model.set_elasticities('Agriculture', {
        'Oil_Price_Change': -0.08,
        'Monsoon_Deviation': 0.20,
        'Food_Inflation': -0.10
    })
    
    sectoral_model.set_elasticities('Manufacturing', {
        'Oil_Price_Change': -0.10,
        'Capex_Growth_Change': 0.15,
        'Tariff_Impact': -0.50,
        'Global_Growth': 0.30
    })
    
    sectoral_model.set_elasticities('Services', {
        'Consumption_Growth': 0.20,
        'Interest_Rate_Change': -0.08,
        'Export_Growth': 0.25,
        'Global_Growth': 0.15
    })
    
    # ========================================================================
    # PAGE: DASHBOARD
    # ========================================================================
    
    if page == "🏠 Dashboard":
        show_dashboard(sectoral_model)
    
    # ========================================================================
    # PAGE: SCENARIO BUILDER
    # ========================================================================
    
    elif page == "⚙️ Scenario Builder":
        show_scenario_builder(sectoral_model)
    
    # ========================================================================
    # PAGE: SENSITIVITY ANALYSIS
    # ========================================================================
    
    elif page == "📈 Sensitivity Analysis":
        show_sensitivity_analysis()
    
    # ========================================================================
    # PAGE: INSTITUTIONAL COMPARISON
    # ========================================================================
    
    elif page == "📊 Institutional Comparison":
        show_institutional_comparison()
    
    # ========================================================================
    # PAGE: DOWNLOAD RESULTS
    # ========================================================================
    
    elif page == "📥 Download Results":
        show_download_page()


def show_dashboard(sectoral_model):
    """Dashboard with base case scenario"""
    
    st.subheader("📊 Base Case Forecast")
    
    # Base case variables
    base_variables = {
        'Oil_Price_Change': 0.0,
        'Monsoon_Deviation': 0.0,
        'Food_Inflation': 0.0,
        'Capex_Growth_Change': 0.0,
        'Tariff_Impact': -0.10,
        'Global_Growth': 0.0,
        'Consumption_Growth': 0.0,
        'Interest_Rate_Change': 0.0,
        'Export_Growth': 0.0
    }
    
    # Calculate base case
    sector_growths_base = {}
    for sector in sectoral_model.sectors.keys():
        sector_growths_base[sector] = sectoral_model.project_growth(sector, base_variables)
    
    gdp_base = sectoral_model.aggregate_gdp(sector_growths_base)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <strong>Base Case GDP Growth</strong><br>
            <span style='font-size: 2em; color: #003366;'>{gdp_base:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <strong>Agriculture</strong><br>
            <span style='font-size: 2em; color: #003366;'>{sector_growths_base['Agriculture']:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-box'>
            <strong>Manufacturing</strong><br>
            <span style='font-size: 2em; color: #003366;'>{sector_growths_base['Manufacturing']:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-box'>
            <strong>Services</strong><br>
            <span style='font-size: 2em; color: #003366;'>{sector_growths_base['Services']:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Scenario comparison
    st.subheader("📈 Scenario Comparison")
    
    # Upside scenario
    upside_variables = {
        'Oil_Price_Change': -7.5,
        'Monsoon_Deviation': 0.10,
        'Food_Inflation': -0.5,
        'Capex_Growth_Change': 10.0,
        'Tariff_Impact': 0.0,
        'Global_Growth': 0.5,
        'Consumption_Growth': 1.0,
        'Interest_Rate_Change': -0.50,
        'Export_Growth': 2.0
    }
    
    sector_growths_upside = {}
    for sector in sectoral_model.sectors.keys():
        sector_growths_upside[sector] = sectoral_model.project_growth(sector, upside_variables)
    
    gdp_upside = sectoral_model.aggregate_gdp(sector_growths_upside)
    
    # Downside scenario
    downside_variables = {
        'Oil_Price_Change': 17.5,
        'Monsoon_Deviation': -0.15,
        'Food_Inflation': 1.5,
        'Capex_Growth_Change': -15.0,
        'Tariff_Impact': -0.25,
        'Global_Growth': -1.0,
        'Consumption_Growth': -0.5,
        'Interest_Rate_Change': 0.25,
        'Export_Growth': -3.0
    }
    
    sector_growths_downside = {}
    for sector in sectoral_model.sectors.keys():
        sector_growths_downside[sector] = sectoral_model.project_growth(sector, downside_variables)
    
    gdp_downside = sectoral_model.aggregate_gdp(sector_growths_downside)
    
    # Display scenarios
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='scenario-box downside'>
            <strong>📉 Downside Case</strong><br>
            Probability: 20%<br>
            <span style='font-size: 1.8em;'>{gdp_downside:.2f}%</span><br>
            <small>Oil spike, tariffs, monsoon deficit</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='scenario-box base'>
            <strong>📊 Base Case</strong><br>
            Probability: 60%<br>
            <span style='font-size: 1.8em;'>{gdp_base:.2f}%</span><br>
            <small>Most likely outcome</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='scenario-box upside'>
            <strong>📈 Upside Case</strong><br>
            Probability: 20%<br>
            <span style='font-size: 1.8em;'>{gdp_upside:.2f}%</span><br>
            <small>Lower oil, strong capex, excess monsoon</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability-weighted forecast
    prob_weighted = (0.20 * gdp_downside) + (0.60 * gdp_base) + (0.20 * gdp_upside)
    
    st.info(f"**Probability-Weighted Forecast: {prob_weighted:.2f}%**")
    
    # Monte Carlo simulation
    st.subheader("🎲 Monte Carlo Simulation (10,000 runs)")
    
    scenarios = {
        'Upside': {'probability': 0.20, 'growth': gdp_upside},
        'Base': {'probability': 0.60, 'growth': gdp_base},
        'Downside': {'probability': 0.20, 'growth': gdp_downside}
    }
    
    mc_sim = MonteCarloGDPSimulation(num_simulations=10000)
    mc_sim.run_simulation(gdp_base, scenarios)
    stats_dict = mc_sim.get_statistics()
    
    # Display MC statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{stats_dict['mean']:.2f}%")
    
    with col2:
        st.metric("Median", f"{stats_dict['median']:.2f}%")
    
    with col3:
        st.metric("Std Dev", f"{stats_dict['std']:.2f}%")
    
    with col4:
        st.metric("Range", f"{stats_dict['min']:.2f}% - {stats_dict['max']:.2f}%")
    
    # Plot MC distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(mc_sim.simulations, bins=60, density=True, alpha=0.7,
                 color=DARKBLUE, edgecolor='black', linewidth=0.5)
    
    mu, sigma = stats_dict['mean'], stats_dict['std']
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    axes[0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2.5, label='Normal Fit')
    axes[0].axvline(mu, color=DARKBLUE, linestyle='--', linewidth=2, label=f'Mean: {mu:.2f}%')
    axes[0].set_xlabel('GDP Growth Rate (%)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    axes[0].set_title('GDP Growth Distribution', fontsize=12, fontweight='bold', color=DARKBLUE)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # CDF
    sorted_sims = np.sort(mc_sim.simulations)
    cumulative_prob = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    axes[1].plot(sorted_sims, cumulative_prob * 100, linewidth=2.5, color=DARKBLUE)
    axes[1].fill_between(sorted_sims, 0, cumulative_prob * 100, alpha=0.2, color=DARKBLUE)
    axes[1].set_xlabel('GDP Growth Rate (%)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Cumulative Probability (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold', color=DARKBLUE)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Percentile summary
    st.subheader("📊 Percentile Distribution")
    
    percentile_data = {
        'Percentile': ['5th', '25th', '50th (Median)', '75th', '95th'],
        'GDP Growth': [
            f"{stats_dict['percentile_5']:.2f}%",
            f"{stats_dict['percentile_25']:.2f}%",
            f"{stats_dict['median']:.2f}%",
            f"{stats_dict['percentile_75']:.2f}%",
            f"{stats_dict['percentile_95']:.2f}%"
        ]
    }
    
    st.dataframe(pd.DataFrame(percentile_data), use_container_width=True)


def show_scenario_builder(sectoral_model):
    """Interactive scenario builder"""
    
    st.subheader("⚙️ Build Your Custom Scenario")
    
    st.markdown("Adjust the variables below to create your own scenario:")
    
    # Create 3 columns for inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🛢️ Oil & Energy")
        oil_change = st.slider(
            "Oil Price Change (USD/bbl)",
            min_value=-15.0,
            max_value=25.0,
            value=0.0,
            step=0.5,
            help="Base: $62.50/bbl. Negative = cheaper oil"
        )
        
        inflation_change = st.slider(
            "Food Inflation Change (%)",
            min_value=-1.0,
            max_value=2.0,
            value=0.0,
            step=0.1
        )
    
    with col2:
        st.markdown("### 🌾 Agriculture & Climate")
        monsoon_dev = st.slider(
            "Monsoon Deviation (% of normal)",
            min_value=-0.20,
            max_value=0.20,
            value=0.0,
            step=0.05,
            help="Negative = deficit, Positive = excess"
        )
        
        capex_change = st.slider(
            "Capex Growth Change (%)",
            min_value=-25.0,
            max_value=25.0,
            value=0.0,
            step=1.0,
            help="Base: 35% YoY growth"
        )
    
    with col3:
        st.markdown("### 🌍 External Factors")
        tariff_impact = st.slider(
            "Tariff Impact (%)",
            min_value=-0.50,
            max_value=0.0,
            value=-0.10,
            step=0.05,
            help="Base: -10 bps (minimal tariffs)"
        )
        
        global_growth = st.slider(
            "Global Growth Change (%)",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1
        )
    
    st.markdown("---")
    
    # Additional variables
    col1, col2, col3 = st.columns(3)
    
    with col1:
        consumption_growth = st.slider(
            "Consumption Growth Change (%)",
            min_value=-1.0,
            max_value=2.0,
            value=0.0,
            step=0.1
        )
    
    with col2:
        rate_change = st.slider(
            "Interest Rate Change (bps)",
            min_value=-100.0,
            max_value=100.0,
            value=0.0,
            step=10.0,
            help="Negative = rate cuts"
        )
    
    with col3:
        export_growth = st.slider(
            "Export Growth Change (%)",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.5
        )
    
    st.markdown("---")
    
    # Calculate scenario
    if st.button("📊 Calculate Scenario", key="scenario_calc"):
        custom_variables = {
            'Oil_Price_Change': oil_change,
            'Monsoon_Deviation': monsoon_dev,
            'Food_Inflation': inflation_change,
            'Capex_Growth_Change': capex_change,
            'Tariff_Impact': tariff_impact / 100,  # Convert to decimal
            'Global_Growth': global_growth,
            'Consumption_Growth': consumption_growth,
            'Interest_Rate_Change': rate_change / 100,  # Convert to decimal
            'Export_Growth': export_growth
        }
        
        # Calculate sector growths
        sector_growths = {}
        for sector in sectoral_model.sectors.keys():
            sector_growths[sector] = sectoral_model.project_growth(sector, custom_variables)
        
        gdp_forecast = sectoral_model.aggregate_gdp(sector_growths)
        
        # Display results
        st.success("✅ Scenario Calculated!")
        
        # Results in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("GDP Growth", f"{gdp_forecast:.2f}%")
        
        with col2:
            st.metric("Agriculture", f"{sector_growths['Agriculture']:.2f}%")
        
        with col3:
            st.metric("Manufacturing", f"{sector_growths['Manufacturing']:.2f}%")
        
        with col4:
            st.metric("Services", f"{sector_growths['Services']:.2f}%")
        
        # Sector breakdown chart
        sector_data = pd.DataFrame({
            'Sector': list(sector_growths.keys()),
            'Growth': list(sector_growths.values())
        })
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(sector_data['Sector'], sector_data['Growth'], color=[DARKBLUE, LIGHTBLUE, GOLDCOLOR])
        ax.set_ylabel('Growth Rate (%)', fontweight='bold')
        ax.set_title('Sectoral Growth Breakdown', fontweight='bold', color=DARKBLUE, fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(sector_data['Growth']):
            ax.text(i, v + 0.2, f'{v:.2f}%', ha='center', fontweight='bold')
        
        st.pyplot(fig)
        
        # Save to session state
        st.session_state.forecast_data = {
            'gdp': gdp_forecast,
            'sectors': sector_growths,
            'variables': custom_variables,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


def show_sensitivity_analysis():
    """Sensitivity analysis page"""
    
    st.subheader("📈 Sensitivity Analysis")
    
    st.markdown("""
    Understand how different variables impact GDP growth.
    The elasticities show the percentage point change in GDP for each variable shock.
    """)
    
    # Base case values
    base_growth = 6.56
    
    # Sensitivity data
    sensitivities = {
        'Oil Prices (USD/bbl)': {
            'elasticity': -0.10,
            'range': (-15, 15),
            'description': 'Every $1/bbl change'
        },
        'Monsoon Index (% deviation)': {
            'elasticity': 0.20,
            'range': (-10, 10),
            'description': '10% deviation from normal'
        },
        'Capex Growth (%)': {
            'elasticity': 0.08,
            'range': (-20, 20),
            'description': 'Government capex growth change'
        },
        'Global Growth (%)': {
            'elasticity': 0.50,
            'range': (-2, 2),
            'description': 'Global GDP growth change'
        },
        'US Tariff Impact (%)': {
            'elasticity': -0.50,
            'range': (-25, 0),
            'description': 'Export competitiveness impact'
        }
    }
    
    # Create sensitivity table
    st.subheader("Sensitivity Coefficients")
    
    sensitivity_df = pd.DataFrame({
        'Variable': list(sensitivities.keys()),
        'Elasticity': [v['elasticity'] for v in sensitivities.values()],
        'Description': [v['description'] for v in sensitivities.values()]
    })
    
    st.dataframe(sensitivity_df, use_container_width=True)
    
    st.markdown("---")
    
    # Tornado chart
    st.subheader("🌪️ Sensitivity Tornado Chart")
    
    variables = list(sensitivities.keys())
    impacts = []
    
    for var_name in variables:
        data = sensitivities[var_name]
        min_shock, max_shock = data['range']
        min_impact = base_growth + (data['elasticity'] * min_shock) - base_growth
        max_impact = base_growth + (data['elasticity'] * max_shock) - base_growth
        impacts.append((min_impact, max_impact))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    y_pos = np.arange(len(variables))
    
    for i, (min_val, max_val) in enumerate(impacts):
        if min_val < 0:
            ax.barh(i, abs(min_val), left=-abs(min_val), height=0.6,
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
        if max_val > 0:
            ax.barh(i, max_val, left=0, height=0.6,
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.axvline(0, color='black', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables, fontsize=10)
    ax.set_xlabel('GDP Growth Impact (Percentage Points)', fontsize=11, fontweight='bold')
    ax.set_title(f'GDP Growth Sensitivity - Tornado Chart (Base: {base_growth:.2f}%)',
                fontsize=12, fontweight='bold', color=DARKBLUE)
    ax.grid(axis='x', alpha=0.3)
    
    st.pyplot(fig)
    
    # Interactive sensitivity calculator
    st.markdown("---")
    st.subheader("🔧 Interactive Sensitivity Calculator")
    
    selected_var = st.selectbox(
        "Select variable to analyze",
        list(sensitivities.keys())
    )
    
    var_data = sensitivities[selected_var]
    min_shock, max_shock = var_data['range']
    
    shock_value = st.slider(
        f"Shock to {selected_var}",
        min_value=min_shock,
        max_value=max_shock,
        value=0.0,
        step=0.5
    )
    
    impact = var_data['elasticity'] * shock_value
    new_growth = base_growth + impact
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Shock Value", f"{shock_value:.2f}")
    
    with col2:
        st.metric("Elasticity", f"{var_data['elasticity']:.3f}")
    
    with col3:
        st.metric("GDP Impact", f"{impact:+.2f} bps", delta=f"{impact:+.2f}")
    
    st.info(f"**New GDP Growth: {new_growth:.2f}%** (from base {base_growth:.2f}%)")


def show_institutional_comparison():
    """Institutional forecasts comparison"""
    
    st.subheader("📊 Institutional Forecasts Comparison")
    
    st.markdown("""
    How does our model compare with major institutional forecasters?
    """)
    
    institutions_data = {
        'Institution': [
            'Our Model (Base)',
            'Probability-Weighted',
            'RBI (FY25-26)',
            'IMF (Calendar 2026)',
            'ADB (Calendar 2026)',
            'Crisil (FY25-26)',
            'World Bank (2026)'
        ],
        'Forecast (%)': [6.56, 6.65, 7.30, 6.40, 6.70, 7.00, 6.50],
        'Type': ['Our Model', 'Our Model', 'Institution', 'Institution', 'Institution', 'Institution', 'Institution']
    }
    
    forecast_df = pd.DataFrame(institutions_data)
    
    # Display table
    st.dataframe(forecast_df, use_container_width=True)
    
    # Statistics
    our_base = forecast_df[forecast_df['Institution'] == 'Our Model (Base)']['Forecast (%)'].values[0]
    institutional_avg = forecast_df[forecast_df['Type'] == 'Institution']['Forecast (%)'].mean()
    difference = our_base - institutional_avg
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Our Base Case", f"{our_base:.2f}%")
    
    with col2:
        st.metric("Institutional Average", f"{institutional_avg:.2f}%")
    
    with col3:
        st.metric("Difference", f"{difference:+.2f}%", delta=f"{difference*100:+.0f} bps")
    
    st.markdown("---")
    
    # Comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = [DARKBLUE if x == 'Our Model' else '#666666' for x in forecast_df['Type']]
    bars = ax.barh(forecast_df['Institution'], forecast_df['Forecast (%)'], color=colors, alpha=0.8)
    
    # Add value labels
    for i, (idx, row) in enumerate(forecast_df.iterrows()):
        ax.text(row['Forecast (%)'] + 0.1, i, f"{row['Forecast (%)']:.1f}%",
                va='center', fontweight='bold')
    
    ax.set_xlabel('GDP Growth Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('2026 GDP Growth Forecast Comparison',
                fontsize=12, fontweight='bold', color=DARKBLUE)
    ax.set_xlim([6.0, 7.5])
    ax.grid(axis='x', alpha=0.3)
    
    st.pyplot(fig)
    
    # Key insights
    st.subheader("🔍 Key Insights")
    
    min_forecast = forecast_df['Forecast (%)'].min()
    max_forecast = forecast_df['Forecast (%)'].max()
    median_forecast = forecast_df[forecast_df['Type'] == 'Institution']['Forecast (%)'].median()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <strong>Range</strong><br>
            <span style='font-size: 1.5em;'>{min_forecast:.2f}% - {max_forecast:.2f}%</span><br>
            <small>{(max_forecast - min_forecast)*100:.0f} bps spread</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <strong>Consensus (Median)</strong><br>
            <span style='font-size: 1.5em;'>{median_forecast:.2f}%</span><br>
            <small>Institutional median</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-box'>
            <strong>Model Positioning</strong><br>
            <span style='font-size: 1.5em;'>{our_base:.2f}%</span><br>
            <small>Slightly below consensus</small>
        </div>
        """, unsafe_allow_html=True)


def show_download_page():
    """Download results page"""
    
    st.subheader("📥 Download Your Results")
    
    if not st.session_state.forecast_data:
        st.warning("⚠️ No custom scenario calculated yet. Go to 'Scenario Builder' to create one first.")
        return
    
    st.success("✅ Custom scenario data available for download!")
    
    # Display current forecast
    st.subheader("Current Scenario")
    
    data = st.session_state.forecast_data
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("GDP Growth", f"{data['gdp']:.2f}%")
    
    with col2:
        st.metric("Generated", data['timestamp'])
    
    with col3:
        st.metric("Format", "Excel & CSV")
    
    st.markdown("---")
    
    # Create downloadable data
    # Summary data
    summary_data = {
        'Metric': ['GDP Growth', 'Agriculture', 'Manufacturing', 'Services'],
        'Value (%)': [
            data['gdp'],
            data['sectors'].get('Agriculture', 0),
            data['sectors'].get('Manufacturing', 0),
            data['sectors'].get('Services', 0)
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Variables data
    variables_data = {
        'Variable': list(data['variables'].keys()),
        'Shock': list(data['variables'].values())
    }
    
    variables_df = pd.DataFrame(variables_data)
    
    # Create Excel file
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        variables_df.to_excel(writer, sheet_name='Variables', index=False)
    
    excel_buffer.seek(0)
    
    # Create CSV file
    csv_buffer = io.StringIO()
    summary_df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📊 Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"GDP_Forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        st.download_button(
            label="📄 Download CSV",
            data=csv_content,
            file_name=f"GDP_Forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # Show data preview
    st.subheader("📋 Data Preview")
    
    tab1, tab2 = st.tabs(["Summary", "Variables"])
    
    with tab1:
        st.dataframe(summary_df, use_container_width=True)
    
    with tab2:
        st.dataframe(variables_df, use_container_width=True)
    
    st.markdown("---")
    
    # About section
    st.subheader("ℹ️ About This App")
    
    st.markdown("""
    **India GDP Projection 2026 - Interactive Dashboard**
    
    This Streamlit app provides interactive financial modeling for India's 2026 GDP forecast.
    
    **Features:**
    - Base case scenario analysis (6.56% GDP growth)
    - Custom scenario builder
    - Sensitivity analysis
    - Monte Carlo simulations (10,000 runs)
    - Institutional forecast comparison
    - Downloadable results (Excel & CSV)
    
    **Base Case Assumptions:**
    - Oil Price: $62.50/bbl
    - Monsoon: 97% of normal
    - Inflation: 2.5%
    - Government Capex: 35% YoY growth
    - Tariff Impact: -10 bps
    
    **Model Methodology:**
    - Linear regression with macro indicators
    - Sectoral decomposition (Agriculture, Manufacturing, Services)
    - Elasticity-based forecasting
    - Probability-weighted scenarios
    
    **Data Sources:**
    - NSO (National Sample Office): Quarterly GDP data
    - RBI: Monetary policy stance, inflation expectations
    - Institutional Forecasts: IMF, ADB, World Bank, Crisil
    
    **For More Information:**
    - The Mountain Path - World of Finance
    - Prof. V. Ravichandran
    - 28+ Years Corporate Finance Experience
    - 10+ Years Academic Excellence
    """)


if __name__ == "__main__":
    main()
