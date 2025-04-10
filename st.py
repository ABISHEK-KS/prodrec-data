import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from streamlit.components.v1 import html
from functools import reduce

# Configurations
st.set_page_config(
    page_title="ProdRec - Smart Product Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS and JavaScript
def inject_custom_style():
    st.markdown("""
    <style>
        /* Base styling */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #ffffff;
        }
        .stApp {
            background-color: #f5f5f5;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #222222;
            font-weight: 600;
        }
        .stButton>button {
            background-color: #222222;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #444444;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .stSelectbox, .stMultiselect {
            border-radius: 8px;
        }
        
        /* Card styling */
        .card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid #e0e0e0;
            transition: all 0.3s ease;
        }
        .card:hover {
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
            transform: translateY(-2px);
        }
        
        /* Comparison step cards */
        .step-card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            padding: 24px;
            margin-bottom: 16px;
            border-left: 4px solid #222222;
        }
        
        /* Graph containers */
        .graph-container {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border: 1px solid #e0e0e0;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade {
            animation: fadeIn 0.6s ease-out forwards;
        }
        
        /* Sidebar */
        .sidebar .sidebar-content {
            background-color: #222222;
            color: white;
        }
        .sidebar .sidebar-content .stSelectbox, 
        .sidebar .sidebar-content .stMultiselect {
            background-color: white;
            color: #222222;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            margin-top: 40px;
            color: #666;
            font-size: 14px;
            border-top: 1px solid #e0e0e0;
        }
        
        /* Progress bar */
        .progress-container {
            width: 100%;
            background-color: #e0e0e0;
            border-radius: 8px;
            margin: 20px 0;
        }
        .progress-bar {
            height: 8px;
            background-color: #222222;
            border-radius: 8px;
            width: 0%;
            transition: width 0.4s ease;
        }
    </style>
    
    <script>
        // Animation on load
        document.addEventListener('DOMContentLoaded', function() {
            const elements = document.querySelectorAll('.card, .graph-container, .stButton');
            elements.forEach((el, index) => {
                setTimeout(() => {
                    el.classList.add('animate-fade');
                }, index * 100);
            });
            
            // Smooth scrolling for comparison steps
            document.querySelectorAll('.comparison-step').forEach(step => {
                step.style.opacity = '0';
                step.style.transform = 'translateY(20px)';
                step.style.transition = 'all 0.6s ease-out';
            });
        });
        
        // Function to animate comparison steps
        function animateComparisonStep(stepNumber) {
            const step = document.querySelector(`.comparison-step-${stepNumber}`);
            if (step) {
                step.style.opacity = '1';
                step.style.transform = 'translateY(0)';
            }
        }
        
        // Function to update progress bar
        function updateProgressBar(percentage) {
            const progressBar = document.querySelector('.progress-bar');
            if (progressBar) {
                progressBar.style.width = `${percentage}%`;
            }
        }
    </script>
    """, unsafe_allow_html=True)

# Data loading functions
@st.cache_data
def load_product_data(folder_path="aadv"):
    """Load product data from CSV files in the specified folder."""
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
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

@st.cache_data
def load_phone_data():
    """Load phone data from CSV file."""
    return pd.read_csv('prp.csv')

def clean_price(price_str):
    """Clean and convert price strings to numeric values."""
    if pd.isna(price_str) or price_str == '':
        return np.nan
    cleaned = re.sub(r'[^\d.]', '', str(price_str))
    try:
        return float(cleaned)
    except ValueError:
        return np.nan

def clean_ratings(rating_str):
    """Clean and convert rating strings to numeric values."""
    try:
        return float(rating_str) if pd.notna(rating_str) else np.nan
    except ValueError:
        return np.nan

# Home Page
def show_home():
    """Display the home page."""
    st.title("🛍️ Welcome to ProdRec")
    st.markdown("### Your Smart Product Comparison Platform")
    
    with st.container():
        st.markdown("""
        <div class='card'>
            <h3 style='color: #222222;'>ProdRec helps you make informed purchasing decisions by:</h3>
            <ul>
                <li>Providing detailed product comparisons</li>
                <li>Visualizing key metrics and features</li>
                <li>Recommending the best options based on your priorities</li>
            </ul>
            <p>Get started by selecting a comparison tool from the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.image(
        "https://images.unsplash.com/photo-1555529669-e69e7aa0ba9a?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
        use_column_width=True,
        caption="Compare. Analyze. Decide."
    )
    
    st.markdown("---")
    st.markdown("""
    <div class='footer'>
        © 2024 ProdRec | Data-Driven Shopping Decisions
    </div>
    """, unsafe_allow_html=True)

# Product Comparison Page
def show_product_comparison():
    """Display the product comparison page."""
    st.title("📊 Product Comparison")
    st.markdown("Compare products across categories and features.")
    
    # Load data
    data = load_product_data()
    if data.empty:
        st.error("No product data available.")
        return
    
    # Clean data
    data['discount_price'] = data['discount_price'].apply(lambda x: clean_price(str(x)))
    data['actual_price'] = data['actual_price'].apply(lambda x: clean_price(str(x)))
    data['ratings'] = data['ratings'].apply(clean_ratings)
    
    # Category selection
    categories = data['category'].dropna().unique()
    selected_category = st.selectbox("Select Category", categories[:10])
    
    # Filter data
    filtered_data = data[data['category'] == selected_category]
    product_options = filtered_data['name'].unique()
    selected_products = st.multiselect("Select Products", product_options, default=product_options[:2])
    
    if len(selected_products) >= 2:
        comparison_data = filtered_data[filtered_data['name'].isin(selected_products)].copy()
        
        # Calculate comparison metrics
        comparison_data['price_points'] = 100 * (1 - (comparison_data['discount_price'] - comparison_data['discount_price'].min()) / 
                                             (comparison_data['discount_price'].max() - comparison_data['discount_price'].min()))
        comparison_data['rating_points'] = 100 * (comparison_data['ratings'] / comparison_data['ratings'].max())
        comparison_data['total_points'] = comparison_data['price_points'] + comparison_data['rating_points']
        
        # Display comparison table
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Comparison Summary")
            st.dataframe(
                comparison_data[['name', 'discount_price', 'actual_price', 'ratings', 'total_points']]
                .rename(columns={
                    'name': 'Product',
                    'discount_price': 'Price',
                    'actual_price': 'Original Price',
                    'ratings': 'Rating',
                    'total_points': 'Score'
                })
                .style.format({
                    'Price': '₹{:.2f}',
                    'Original Price': '₹{:.2f}',
                    'Rating': '{:.1f}',
                    'Score': '{:.1f}'
                })
                .background_gradient(cmap='Greys')
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Best product
        best_product = comparison_data.loc[comparison_data['total_points'].idxmax()]
        
        # Step-by-step comparison
        st.subheader("Step-by-Step Comparison")
        st.markdown("""
        <div class='progress-container'>
            <div class='progress-bar' id='progress-bar'></div>
        </div>
        """, unsafe_allow_html=True)
        
        steps = [
            ("Price Comparison", "discount_price", "Lower price is better", "₹{:.2f}", False),
            ("Rating Comparison", "ratings", "Higher rating is better", "{:.1f}", True),
            ("Value for Money", "total_points", "Higher score indicates better overall value", "{:.1f}", True)
        ]
        
        for i, (title, col, desc, fmt, higher_better) in enumerate(steps, 1):
            with st.container():
                st.markdown(f"""
                <div class='step-card comparison-step comparison-step-{i}'>
                    <h3>{i}. {title}</h3>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Update progress bar
                html(f"""
                <script>
                    updateProgressBar({i / len(steps) * 100});
                    setTimeout(() => animateComparisonStep({i}), 200);
                </script>
                """)
                
                # Create figure
                fig = go.Figure()
                
                for _, row in comparison_data.iterrows():
                    value = row[col]
                    formatted_value = fmt.format(value)
                    color = '#222222' if row['name'] == best_product['name'] else '#888888'
                    
                    fig.add_trace(go.Bar(
                        x=[row['name']],
                        y=[value],
                        name=row['name'],
                        text=[formatted_value],
                        textposition='auto',
                        marker_color=color,
                        hovertemplate=f"<b>{row['name']}</b><br>{title}: {formatted_value}<extra></extra>"
                    ))
                
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis_title="Product",
                    yaxis_title=title,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
        
        # Final recommendation
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🏆 Final Recommendation")
            st.markdown(f"""
            <div style='background: #f5f5f5; padding: 20px; border-radius: 8px;'>
                <h3 style='color: #222222;'>{best_product['name']}</h3>
                <p><strong>Score:</strong> {best_product['total_points']:.1f}/200</p>
                <p><strong>Price:</strong> ₹{best_product['discount_price']:.2f}</p>
                <p><strong>Rating:</strong> {best_product['ratings']:.1f}</p>
                <p>This product offers the best balance of price and quality in your selection.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Additional visualizations
        with st.container():
            st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
            st.subheader("Price vs Rating Analysis")
            
            fig = px.scatter(
                comparison_data,
                x='discount_price',
                y='ratings',
                size='total_points',
                color='name',
                color_discrete_sequence=['#222222', '#666666', '#999999'],
                hover_name='name',
                labels={
                    'discount_price': 'Price (₹)',
                    'ratings': 'Rating',
                    'total_points': 'Overall Score'
                }
            )
            
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#e0e0e0'),
                yaxis=dict(showgrid=True, gridcolor='#e0e0e0'),
                legend_title_text='Product'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div style='padding: 15px; background: #f5f5f5; border-radius: 8px; margin-top: 15px;'>
                <strong>Insight:</strong> This scatter plot shows the relationship between price and ratings. 
                The ideal products are in the top-left quadrant (low price, high rating).
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Radar chart
        with st.container():
            st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
            st.subheader("Feature Radar Chart")
            
            categories = ['Price', 'Rating', 'Value']
            fig = go.Figure()
            
            for _, row in comparison_data.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[
                        1 - (row['discount_price'] - comparison_data['discount_price'].min()) / 
                        (comparison_data['discount_price'].max() - comparison_data['discount_price'].min()),
                        row['ratings'] / comparison_data['ratings'].max(),
                        row['total_points'] / comparison_data['total_points'].max()
                    ],
                    theta=categories,
                    fill='toself',
                    name=row['name'],
                    line_color='#222222' if row['name'] == best_product['name'] else '#888888'
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1]),
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div style='padding: 15px; background: #f5f5f5; border-radius: 8px; margin-top: 15px;'>
                <strong>Insight:</strong> The radar chart provides a quick visual comparison of key metrics. 
                Larger area indicates better overall performance.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.warning("Please select at least 2 products for comparison.")
    
    st.markdown("---")
    st.markdown("""
    <div class='footer'>
        © 2024 ProdRec | Data-Driven Shopping Decisions
    </div>
    """, unsafe_allow_html=True)

# Phone Comparison Page
def show_phone_comparison():
    """Display the phone comparison page."""
    st.title("📱 Smartphone Comparison")
    st.markdown("Compare smartphones based on specifications and features.")
    
    # Load data
    phone_data = load_phone_data()
    
    # Brand selection
    brands = phone_data['Brand'].unique()
    selected_brands = st.multiselect("Select Brands", brands, default=brands[:1])
    
    # Model selection
    selected_models = []
    for brand in selected_brands:
        brand_models = phone_data[phone_data['Brand'] == brand]['Model'].unique()
        selected_model = st.selectbox(f"Select {brand} Model", brand_models)
        selected_models.append(selected_model)
    
    if len(selected_models) >= 2:
        # Filter data
        comparison_data = phone_data[phone_data['Model'].isin(selected_models)].copy()
        
        # Clean and convert data
        comparison_data['Memory'] = comparison_data['Memory'].str.replace('GB', '').astype(int)
        comparison_data['Storage'] = comparison_data['Storage'].str.replace('GB', '').astype(int)
        
        # Calculate scores
        comparison_data['Performance Score'] = (
            comparison_data['Memory'] * 0.4 + 
            comparison_data['Storage'] * 0.3 +
            comparison_data['Rating'] * 20
        )
        
        comparison_data['Value Score'] = (
            (100000 / comparison_data['Selling Price']) * 0.6 +
            comparison_data['Rating'] * 0.4
        )
        
        comparison_data['Total Score'] = (
            comparison_data['Performance Score'] * 0.6 +
            comparison_data['Value Score'] * 0.4
        )
        
        # Best phone
        best_phone = comparison_data.loc[comparison_data['Total Score'].idxmax()]
        
        # Display comparison table
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Specification Comparison")
            
            display_cols = [
                'Brand', 'Model', 'Memory', 'Storage', 'Rating', 
                'Selling Price', 'Original Price', 'Total Score'
            ]
            
            st.dataframe(
                comparison_data[display_cols]
                .rename(columns={
                    'Memory': 'RAM (GB)',
                    'Storage': 'Storage (GB)',
                    'Rating': 'Rating (5)',
                    'Selling Price': 'Price (₹)',
                    'Original Price': 'Original (₹)',
                    'Total Score': 'Score'
                })
                .style.format({
                    'Price (₹)': '₹{:.2f}',
                    'Original (₹)': '₹{:.2f}',
                    'Score': '{:.1f}',
                    'Rating (5)': '{:.1f}'
                })
                .background_gradient(cmap='Greys')
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Step-by-step comparison
        st.subheader("Feature Breakdown")
        
        features = [
            ("Performance", "Performance Score", ["RAM (GB)", "Storage (GB)", "Rating"], "Higher is better", "{:.1f}", True),
            ("Value for Money", "Value Score", ["Price (₹)", "Rating"], "Lower price with good rating is best", "{:.1f}", True),
            ("Overall Recommendation", "Total Score", ["Performance Score", "Value Score"], "Balanced performance and value", "{:.1f}", True)
        ]
        
        for i, (title, col, components, desc, fmt, higher_better) in enumerate(features, 1):
            with st.container():
                st.markdown(f"""
                <div class='step-card comparison-step comparison-step-{i}'>
                    <h3>{i}. {title}</h3>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Update progress bar
                html(f"""
                <script>
                    updateProgressBar({i / len(features) * 100});
                    setTimeout(() => animateComparisonStep({i}), 200);
                </script>
                """)
                
                # Create figure
                fig = go.Figure()
                
                for _, row in comparison_data.iterrows():
                    value = row[col]
                    formatted_value = fmt.format(value)
                    color = '#222222' if row['Model'] == best_phone['Model'] else '#888888'
                    
                    fig.add_trace(go.Bar(
                        x=[row['Model']],
                        y=[value],
                        name=row['Model'],
                        text=[formatted_value],
                        textposition='auto',
                        marker_color=color,
                        hovertemplate=f"<b>{row['Model']}</b><br>{title}: {formatted_value}<extra></extra>"
                    ))
                
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis_title="Model",
                    yaxis_title=title,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show component breakdown
                with st.expander(f"See {title} components"):
                    component_fig = go.Figure()
                    
                    for component in components:
                        for _, row in comparison_data.iterrows():
                            component_fig.add_trace(go.Bar(
                                x=[row['Model']],
                                y=[row[component.replace(' (₹)', '').replace(' (GB)', '').replace(' (5)', '')]],
                                name=component,
                                textposition='auto',
                                marker_color='#222222',
                                hovertemplate=f"<b>{row['Model']}</b><br>{component}: {row[component.replace(' (₹)', '').replace(' (GB)', '').replace(' (5)', '')]}<extra></extra>"
                            ))
                    
                    component_fig.update_layout(
                        barmode='group',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        xaxis_title="Model",
                        yaxis_title="Value",
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    
                    st.plotly_chart(component_fig, use_container_width=True)
                
                st.markdown("---")
        
        # Final recommendation
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🏆 Best Overall Phone")
            st.markdown(f"""
            <div style='background: #f5f5f5; padding: 20px; border-radius: 8px;'>
                <h3 style='color: #222222;'>{best_phone['Brand']} {best_phone['Model']}</h3>
                <p><strong>Total Score:</strong> {best_phone['Total Score']:.1f}</p>
                <p><strong>RAM:</strong> {best_phone['Memory']}GB</p>
                <p><strong>Storage:</strong> {best_phone['Storage']}GB</p>
                <p><strong>Price:</strong> ₹{best_phone['Selling Price']:.2f}</p>
                <p><strong>Rating:</strong> {best_phone['Rating']:.1f}/5</p>
                <p>This phone offers the best combination of performance and value in your selection.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Parallel coordinates plot
        with st.container():
            st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
            st.subheader("Feature Comparison")
            
            dimensions = [
                dict(range=[comparison_data['Memory'].min(), comparison_data['Memory'].max()], 
                     label='RAM (GB)', values=comparison_data['Memory']),
                dict(range=[comparison_data['Storage'].min(), comparison_data['Storage'].max()], 
                     label='Storage (GB)', values=comparison_data['Storage']),
                dict(range=[comparison_data['Rating'].min(), comparison_data['Rating'].max()], 
                     label='Rating', values=comparison_data['Rating']),
                dict(range=[comparison_data['Selling Price'].min(), comparison_data['Selling Price'].max()], 
                     label='Price (₹)', values=comparison_data['Selling Price']),
                dict(range=[comparison_data['Total Score'].min(), comparison_data['Total Score'].max()], 
                     label='Total Score', values=comparison_data['Total Score'])
            ]
            
            fig = go.Figure(go.Parcoords(
                line=dict(
                    color=comparison_data['Total Score'],
                    colorscale='Greys',
                    showscale=True,
                    reversescale=True
                ),
                dimensions=dimensions
            ))
            
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=60, r=60, t=40, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div style='padding: 15px; background: #f5f5f5; border-radius: 8px; margin-top: 15px;'>
                <strong>Insight:</strong> This parallel coordinates plot shows how each phone compares across multiple dimensions. 
                Follow the lines to see each phone's strengths and weaknesses.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.warning("Please select at least 2 phones for comparison.")
    
    st.markdown("---")
    st.markdown("""
    <div class='footer'>
        © 2024 ProdRec | Data-Driven Shopping Decisions
    </div>
    """, unsafe_allow_html=True)

# Main App
def main():
    inject_custom_style()
    
    # Sidebar navigation
    st.sidebar.title("ProdRec")
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio("", ["Home", "Product Comparison", "Phone Comparison"])
    
    # Page routing
    if page == "Home":
        show_home()
    elif page == "Product Comparison":
        show_product_comparison()
    elif page == "Phone Comparison":
        show_phone_comparison()

if __name__ == "__main__":
    main()
