import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import re
from streamlit.components.v1 import html

# Configurations
st.set_page_config(
    page_title="ProdRec - Dark Mode",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# BLACK BACKGROUND CSS
def inject_black_theme():
    st.markdown("""
    <style>
        /* FULL BLACK BACKGROUND */
        .main, .stApp, [class*="css"] {
            background-color: #000000 !important;
            color: white !important;
        }
        
        /* TEXT COLOR */
        h1, h2, h3, h4, h5, h6, p, li, .stMarkdown {
            color: white !important;
        }
        
        /* CARDS - DARK GRAY */
        .card {
            background-color: #1a1a1a !important;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #333333;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        
        /* SIDEBAR - DARKER GRAY */
        .sidebar .sidebar-content {
            background-color: #121212 !important;
            border-right: 1px solid #333333;
        }
        
        /* BUTTONS - ACCENT COLOR */
        .stButton>button {
            background-color: #4a00e0 !important;
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            background-color: #6a1b9a !important;
            transform: translateY(-2px);
        }
        
        /* SLIDERS AND INPUTS */
        .stSlider, .stSelectbox, .stMultiselect {
            background-color: #1a1a1a !important;
            border: 1px solid #333333 !important;
        }
        
        /* GRAPH CONTAINERS */
        .graph-container {
            background-color: #1a1a1a !important;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid #333333;
        }
        
        /* PROGRESS BAR */
        .progress-container {
            width: 100%;
            background-color: #333333;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .progress-bar {
            height: 8px;
            background: linear-gradient(90deg, #4a00e0, #8e2de2);
            border-radius: 10px;
            width: 0%;
            transition: width 0.4s ease;
        }
        
        /* WINDOWED COMPARISON */
        .comparison-window {
            background-color: #1a1a1a;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid #4a00e0;
            display: none;
        }
        
        /* ANIMATIONS */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fade {
            animation: fadeIn 0.6s ease-out forwards;
        }
    </style>
    
    <script>
        // WINDOWED COMPARISON SYSTEM
        let currentWindow = 0;
        const totalWindows = 3; // Price, Rating, Final
        
        function showWindow(index) {
            document.querySelectorAll('.comparison-window').forEach((win, i) => {
                win.style.display = i === index ? 'block' : 'none';
            });
            document.getElementById('window-progress').style.width = `${(index+1)/totalWindows*100}%`;
            currentWindow = index;
        }
        
        function nextWindow() {
            if (currentWindow < totalWindows-1) showWindow(currentWindow+1);
        }
        
        function prevWindow() {
            if (currentWindow > 0) showWindow(currentWindow-1);
        }
        
        // Initialize first window
        document.addEventListener('DOMContentLoaded', function() {
            showWindow(0);
        });
    </script>
    """, unsafe_allow_html=True)

# DATA LOADING (UNCHANGED FROM YOUR ORIGINAL)
@st.cache_data
def load_product_data(folder_path="aadv"):
    all_data = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            try:
                data = pd.read_csv(file_path)
                data['category'] = file.replace('.csv', '')
                all_data.append(data)
            except Exception as e:
                st.warning(f"Error reading {file}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

@st.cache_data
def load_phone_data():
    return pd.read_csv('prp.csv')

def clean_price(price_str):
    if pd.isna(price_str) or price_str == '': return np.nan
    cleaned = re.sub(r'[^\d.]', '', str(price_str))
    try: return float(cleaned)
    except ValueError: return np.nan

def clean_ratings(rating_str):
    try: return float(rating_str) if pd.notna(rating_str) else np.nan
    except ValueError: return np.nan

# WINDOWED COMPARISON FUNCTION
def render_windowed_comparison(comparison_data):
    """The window-by-window comparison feature you requested"""
    st.markdown("""
    <div class='progress-container'>
        <div class='progress-bar' id='window-progress'></div>
    </div>
    """, unsafe_allow_html=True)
    
    # WINDOW 1: PRICE COMPARISON
    with st.container():
        st.markdown("""
        <div class='comparison-window'>
            <h3>💰 Price Comparison</h3>
            <p>Analyzing cost-effectiveness of selected products</p>
        """, unsafe_allow_html=True)
        
        # Price comparison chart
        fig = go.Figure()
        for _, row in comparison_data.iterrows():
            fig.add_trace(go.Bar(
                x=[row['name']],
                y=[row['discount_price']],
                marker_color='#8e2de2',
                text=[f"₹{row['discount_price']:.2f}"],
                textposition='auto'
            ))
        
        fig.update_layout(
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, title='Price (₹)')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
            <div style='display: flex; justify-content: space-between; margin-top: 20px;'>
                <button onclick='prevWindow()' disabled style='opacity: 0.5;'>Previous</button>
                <button onclick='nextWindow()'>Next: Ratings</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # WINDOW 2: RATING COMPARISON
    with st.container():
        st.markdown("""
        <div class='comparison-window'>
            <h3>⭐ Rating Comparison</h3>
            <p>Evaluating product quality through user ratings</p>
        """, unsafe_allow_html=True)
        
        # Rating comparison chart
        fig = go.Figure()
        for _, row in comparison_data.iterrows():
            fig.add_trace(go.Bar(
                x=[row['name']],
                y=[row['ratings']],
                marker_color='#4a00e0',
                text=[f"{row['ratings']:.1f}/5"],
                textposition='auto'
            ))
        
        fig.update_layout(
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, title='Rating')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
            <div style='display: flex; justify-content: space-between; margin-top: 20px;'>
                <button onclick='prevWindow()'>Previous: Price</button>
                <button onclick='nextWindow()'>Next: Final Verdict</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # WINDOW 3: FINAL VERDICT
    with st.container():
        best_product = comparison_data.loc[comparison_data['ratings'].idxmax()]
        st.markdown(f"""
        <div class='comparison-window'>
            <h3>🏆 Final Verdict</h3>
            <p>Based on comprehensive analysis of your selection</p>
            
            <div class='card' style='background: linear-gradient(135deg, #1a1a1a, #4a00e0) !important;'>
                <h2>{best_product['name']}</h2>
                <p>⭐ Rating: {best_product['ratings']:.1f}/5</p>
                <p>💰 Price: ₹{best_product['discount_price']:.2f}</p>
                <p>This product offers the best value in its category</p>
            </div>
            
            <div style='display: flex; justify-content: space-between; margin-top: 20px;'>
                <button onclick='prevWindow()'>Previous: Ratings</button>
                <button onclick='nextWindow()' disabled style='opacity: 0.5;'>Next</button>
            </div>
        </div>
        """, unsafe_allow_html=True)

# PRODUCT COMPARISON PAGE (NOW WITH WINDOWED FEATURE)
def show_product_comparison():
    st.title("🔍 Product Comparison")
    st.markdown("Compare products with our window-by-window analysis")
    
    data = load_product_data()
    if data.empty:
        st.error("No data available")
        return
    
    # Clean data
    data['discount_price'] = data['discount_price'].apply(lambda x: clean_price(str(x)))
    data['ratings'] = data['ratings'].apply(clean_ratings)
    
    # Selection UI
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        categories = data['category'].dropna().unique()
        selected_category = st.selectbox("Select Category", categories)
        
        filtered_data = data[data['category'] == selected_category]
        selected_products = st.multiselect(
            "Select Products (2-4)", 
            filtered_data['name'].unique(),
            default=filtered_data['name'].iloc[:2]
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    if len(selected_products) >= 2:
        comparison_data = filtered_data[filtered_data['name'].isin(selected_products)].copy()
        
        # WINDOWED COMPARISON SYSTEM
        st.markdown("## Windowed Comparison")
        render_windowed_comparison(comparison_data)
        
        # ADDITIONAL VISUALIZATIONS
        with st.container():
            st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
            st.markdown("### 📊 Price vs Rating Analysis")
            
            fig = px.scatter(
                comparison_data,
                x='discount_price',
                y='ratings',
                size=np.abs(comparison_data['discount_price']),
                color='name',
                color_discrete_sequence=px.colors.qualitative.Dark24,
                hover_name='name',
                labels={'discount_price': 'Price (₹)', 'ratings': 'Rating'}
            )
            
            fig.update_layout(
                plot_bgcolor='#1a1a1a',
                paper_bgcolor='#1a1a1a',
                font=dict(color='white'),
                xaxis=dict(showgrid=False, gridcolor='#333333'),
                yaxis=dict(showgrid=False, gridcolor='#333333')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.warning("Select at least 2 products")

# MAIN APP
def main():
    inject_black_theme()
    
    st.sidebar.title("ProdRec")
    st.sidebar.markdown("""
    <div style='border-left: 3px solid #4a00e0; padding-left: 10px; margin: 20px 0;'>
        <p>Black background comparison tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio("Navigation", ["Home", "Product Comparison"])
    
    if page == "Home":
        st.title("🛒 ProdRec")
        st.markdown("""
        <div class='card'>
            <h2>Product Comparison Tool</h2>
            <p>Now with window-by-window analysis</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        show_product_comparison()

if __name__ == "__main__":
    main()
