import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import re
import numpy as np
from streamlit.components.v1 import html

# Configurations
st.set_page_config(page_title="ProdRec - Product Comparison Platform", page_icon="🛍️", layout="wide")

# Custom CSS and JavaScript for animations
def load_css_js():
    custom_css = """
    <style>
        /* Main styling */
        .main {
            background-color: #f8f9fa;
        }
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .stSidebar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
        }
        .sidebar .sidebar-content {
            background-color: #4a6bdf;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
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
            box-shadow: 0 6px 16px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.15);
        }
        /* Animation for loading */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade {
            animation: fadeIn 0.6s ease-out forwards;
        }
        /* Graph container styling */
        .graph-container {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .graph-explanation {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-top: 15px;
            border-radius: 0 8px 8px 0;
            font-size: 14px;
            color: #555;
        }
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 20px;
            margin-top: 40px;
            color: #666;
            font-size: 14px;
        }
    </style>
    """
    
    custom_js = """
    <script>
        // Simple animation on page load
        document.addEventListener('DOMContentLoaded', function() {
            const elements = document.querySelectorAll('.stButton, .stSelectbox, .card, .graph-container');
            elements.forEach((el, index) => {
                setTimeout(() => {
                    el.classList.add('animate-fade');
                }, index * 100);
            });
        });
        
        // Smooth scrolling for better UX
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
    </script>
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)
    html(custom_js)

# Helper Functions
@st.cache_data
def load_data(file_path):
    """Load dataset from a local CSV file and return a copy to avoid mutation."""
    data = pd.read_csv(file_path)
    
    # Check if 'category' column exists
    if 'category' not in data.columns:
        raise ValueError("'category' column not found in the dataset.")
    
    # Parse categories
    data = parse_categories(data)
    
    return data

def parse_categories(data):
    """Parse categories with synonyms separated by '|'."""
    data['parsed_category'] = data['category'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else 'Unknown')
    return data

def show_home():
    """Render the Home Page."""
    st.title("Welcome to **ProdRec**")
    st.subheader("Your Trusted Companion in Smarter Shopping Decisions")
    
    with st.container():
        st.markdown("""
        <div class='card'>
            <h3 style='color: #2c3e50;'>At <strong>ProdRec</strong>, we simplify your shopping journey by helping you:</h3>
            <ul>
                <li>Discover products tailored to your needs and budget</li>
                <li>Compare features, prices, and ratings side-by-side</li>
                <li>Decide with confidence, backed by clear, transparent insights</li>
            </ul>
            <p>Let's make your shopping experience smarter, easier, and more rewarding.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.image(
        "https://images.unsplash.com/photo-1555529669-e69e7aa0ba9a?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
        use_column_width=True,
        caption="Explore. Compare. Decide."
    )
    
    st.markdown("---")
    
    with st.container():
        st.markdown("### Ready to explore?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Explore Products", key="explore_btn"):
                st.success("Switch to 'Compare Products' using the sidebar.")
        with col2:
            if st.button("Learn More", key="learn_btn"):
                st.info("ProdRec makes smarter shopping possible with data-driven recommendations.")
    
    st.markdown("---")
    st.markdown("""
    <div class='footer'>
        © 2024 ProdRec Inc. | Built with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)

def clean_price(price_str):
    """Function to clean non-numeric characters from price strings and handle empty values."""
    if not price_str or price_str == '' or pd.isna(price_str):
        return np.nan  # Return NaN for empty or invalid price strings
    
    # Ensure price_str is a string and remove non-numeric characters (except dot for decimal)
    cleaned_price = re.sub(r'[^\d.]', '', str(price_str))
    
    try:
        return float(cleaned_price)  # Attempt to convert cleaned price to float
    except ValueError:
        return np.nan  # Return NaN if conversion fails

def load_data_from_folder(folder_path="aadv"):
    """Loads the product data from the 'aadv' folder and returns a DataFrame."""
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
        full_data = pd.concat(all_data, ignore_index=True)
        return full_data
    else:
        st.warning("No valid CSV files found in the folder.")
        return pd.DataFrame()

def clean_ratings(rating_str):
    """Function to clean non-numeric characters from ratings and handle empty values."""
    try:
        return float(rating_str) if pd.notna(rating_str) else np.nan
    except ValueError:
        return np.nan

def show_compare():
    """Render the Product Comparison Page with Visualizations"""
    st.title("🔍 Product Comparison")
    st.markdown("Compare products based on **category**, **price**, **ratings**, and other features.")

    # Load dataset with caching from the 'aadv' folder
    folder_path = "aadv"
    data = load_data_from_folder(folder_path)

    if data.empty:
        st.error("No product data available for comparison.")
        return

    # Clean the data
    data['discount_price'] = data['discount_price'].apply(lambda x: clean_price(str(x)))
    data['actual_price'] = data['actual_price'].apply(lambda x: clean_price(str(x)))
    data['ratings'] = data['ratings'].apply(lambda x: clean_ratings(x))
    data['discount_price'] = pd.to_numeric(data['discount_price'], errors='coerce')
    data['actual_price'] = pd.to_numeric(data['actual_price'], errors='coerce')

    # Category selection
    unique_categories = data['category'].dropna().unique()
    selected_category = st.selectbox("Select a Category", unique_categories[:10])

    # Filter data based on selected category
    filtered_data = data[data['category'] == selected_category]

    # Product selection
    st.subheader("Select Products to Compare")
    product_options = filtered_data['name'].unique()
    selected_products = st.multiselect("Choose Products:", product_options, default=product_options[:2])

    if len(selected_products) >= 2:
        comparison_data = filtered_data[filtered_data['name'].isin(selected_products)]

        # Point-Based Comparison Table
        st.subheader("Point-Based Comparison Table")
        
        # Compute points
        min_price = comparison_data['discount_price'].min()
        comparison_data['price_points'] = comparison_data['discount_price'].apply(lambda x: (1 / (x - min_price + 1)) * 100 if pd.notna(x) else 0)
        max_rating = comparison_data['ratings'].max() if not comparison_data['ratings'].isnull().all() else 1
        comparison_data['rating_points'] = comparison_data['ratings'].apply(lambda x: (x / max_rating) * 100 if pd.notna(x) else 0)
        comparison_data['total_points'] = comparison_data['price_points'] + comparison_data['rating_points']

        # Display the points data in a card
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            points_data = comparison_data[['name', 'price_points', 'rating_points', 'total_points']]
            st.dataframe(points_data.style.background_gradient(cmap='Blues'))
            st.markdown("</div>", unsafe_allow_html=True)

        # Recommendation
        best_product = points_data.loc[points_data['total_points'].idxmax()]
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Recommended Product")
            st.markdown(f"""
            <div style='background: #f0f4ff; padding: 15px; border-radius: 8px;'>
                <h4 style='color: #2c3e50;'>🎯 Best Choice: <strong>{best_product['name']}</strong></h4>
                <p>Total Points: <strong>{best_product['total_points']:.1f}</strong></p>
                <p>This product offers the best balance of price and ratings in your selection.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Visualizations with explanations
        with st.container():
            st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
            
            # 1. Bar Chart for Price and Rating Comparison
            st.subheader("Price vs Ratings Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(comparison_data['name'], comparison_data['discount_price'], alpha=0.7, label='Discount Price', color='#667eea')
            ax.bar(comparison_data['name'], comparison_data['ratings'] * 10, alpha=0.7, label='Ratings x10', color='#764ba2')
            ax.set_xlabel('Product')
            ax.set_ylabel('Value')
            ax.set_title('Price vs Ratings for Selected Products')
            ax.legend()
            st.pyplot(fig)
            
            st.markdown("""
            <div class='graph-explanation'>
                <strong>Explanation:</strong> This bar chart compares the discount price (blue) and ratings (purple, scaled by 10 for better visualization) 
                of the selected products. Lower prices and higher ratings are generally better.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
            # 2. Scatter Plot for Price vs Rating
            st.subheader("Scatter Plot of Price vs Rating")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(comparison_data['discount_price'], comparison_data['ratings'], color='#4facfe', s=100)
            ax.set_xlabel('Discount Price')
            ax.set_ylabel('Ratings')
            ax.set_title('Price vs Rating Scatter Plot')
            st.pyplot(fig)
            
            st.markdown("""
            <div class='graph-explanation'>
                <strong>Explanation:</strong> The scatter plot shows the relationship between price and ratings. 
                Products in the top-left corner (low price, high rating) offer the best value.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
            # 3. Radar Chart for Product Comparison
            st.subheader("Radar Chart: Price vs Ratings")
            categories = ['Discount Price', 'Ratings']
            values = [comparison_data['discount_price'].mean(), comparison_data['ratings'].mean()]
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.plot(categories, values, linewidth=2, linestyle='solid', color='#667eea')
            ax.fill(categories, values, alpha=0.25, color='#764ba2')
            ax.set_title('Radar Chart of Price and Rating Comparison')
            st.pyplot(fig)
            
            st.markdown("""
            <div class='graph-explanation'>
                <strong>Explanation:</strong> The radar chart provides a quick visual comparison of the average price 
                and ratings across selected products. The larger the area covered, the better the overall performance.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("Please select at least 2 products for comparison.")

    st.markdown("---")
    st.markdown("""
    <div class='footer'>
        © 2024 ProdRec Inc. | Built with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)

# Load the phone data
phone_data = pd.read_csv('prp.csv')

def show_phone_comparison():
    """Render Phone Comparison Section with Advanced and Aesthetically Pleasing Visuals."""
    st.title("📱 Compare Phones")
    st.markdown("Compare smartphones based on specifications, ratings, and prices.")

    # Step 1: Select Phones for Comparison (Multiple Brands)
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        brands = phone_data['Brand'].unique()
        selected_brands = st.multiselect("Select Brands", brands, default=brands[:1], key="brand_selector")
        st.markdown("</div>", unsafe_allow_html=True)

    # Prepare list to store selected phones
    selected_phones = []

    for brand in selected_brands:
        brand_data = phone_data[phone_data['Brand'] == brand]
        models = brand_data['Model'].unique()
        selected_model = st.selectbox(f"Select Phone from {brand}", models, key=f"model_selector_{brand}")
        selected_phones.append(selected_model)

    if len(selected_phones) >= 2:
        # Filter the data for selected models
        comparison_data = phone_data[phone_data['Model'].isin(selected_phones)]

        # Clean and convert data
        comparison_data['Memory'] = comparison_data['Memory'].str.replace('GB', '').astype(int)
        comparison_data['Storage'] = comparison_data['Storage'].str.replace('GB', '').astype(int)

        # Assign points for each feature
        def assign_points(phone):
            points = 0
            points += phone['Memory']  # Points for Memory (1 point per GB of RAM)
            points += phone['Storage'] / 64  # Points for Storage (1 point per 64 GB of Storage)
            points += phone['Rating'] * 2  # Points for Rating (Maximum 10 points for a 5-star rating)
            points += 100000 / phone['Selling Price']  # Inverse scoring: lower price gets more points
            return points

        comparison_data['Points'] = comparison_data.apply(assign_points, axis=1)

        # Display comparison table
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Phone Comparison Table")
            table_data = comparison_data[['Brand', 'Model', 'Color', 'Memory', 'Storage', 'Rating', 'Selling Price', 'Original Price', 'Points']]
            table_data = table_data.rename(columns={
                'Brand': 'Brand',
                'Model': 'Model',
                'Color': 'Color',
                'Memory': 'Memory (GB)',
                'Storage': 'Storage (GB)',
                'Rating': 'Rating',
                'Selling Price': 'Selling Price (₹)',
                'Original Price': 'Original Price (₹)',
                'Points': 'Points'
            })
            st.dataframe(table_data.style.background_gradient(cmap='Blues'))
            st.markdown("</div>", unsafe_allow_html=True)

        # Phone Recommendation
        best_phone = comparison_data.loc[comparison_data['Points'].idxmax()]
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 📱 **Recommended Phone with the Best Features**")
            st.markdown(f"""
            <div style='background: #f0f4ff; padding: 15px; border-radius: 8px;'>
                <h4 style='color: #2c3e50;'>🏆 Best Overall: <strong>{best_phone['Model']}</strong> by {best_phone['Brand']}</h4>
                <p><strong>Total Points:</strong> {best_phone['Points']:.1f}</p>
                <p><strong>Selling Price:</strong> ₹{best_phone['Selling Price']}</p>
                <p><strong>Rating:</strong> {best_phone['Rating']} stars</p>
                <p>This phone offers the most balanced combination of <strong>features, performance</strong>, and <strong>price</strong>!</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Visualizations with explanations
        with st.container():
            st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
            st.subheader("Rating Distribution by Model (Violin Plot)")
            fig = px.violin(comparison_data, y='Rating', x='Model', color='Model', box=True, points="all", 
                          title="Rating Distribution by Model", color_discrete_sequence=['#667eea', '#764ba2'])
            st.plotly_chart(fig)
            
            st.markdown("""
            <div class='graph-explanation'>
                <strong>Explanation:</strong> The violin plot shows the distribution of ratings for each phone model. 
                The wider sections represent where most ratings are concentrated, while the thinner sections show less common ratings.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
            st.subheader("Selling Price Distribution by Model (Violin Plot)")
            fig = px.violin(comparison_data, y='Selling Price', x='Model', color='Model', box=True, points="all", 
                          title="Selling Price Distribution by Model", color_discrete_sequence=['#667eea', '#764ba2'])
            st.plotly_chart(fig)
            
            st.markdown("""
            <div class='graph-explanation'>
                <strong>Explanation:</strong> This violin plot visualizes the price distribution for each model. 
                The white dot represents the median price, while the thick black bar shows the interquartile range.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
            st.subheader("Memory (RAM) Distribution by Model (Violin Plot)")
            fig = px.violin(comparison_data, y='Memory', x='Model', color='Model', box=True, points="all", 
                          title="Memory (RAM) Distribution by Model", color_discrete_sequence=['#667eea', '#764ba2'])
            st.plotly_chart(fig)
            
            st.markdown("""
            <div class='graph-explanation'>
                <strong>Explanation:</strong> This plot shows the RAM distribution for each phone model. 
                Higher RAM typically means better performance for multitasking and demanding apps.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
            st.subheader("Parallel Coordinates Plot")
            fig = px.parallel_coordinates(
                comparison_data, color="Points", dimensions=["Memory", "Storage", "Rating", "Selling Price", "Points"],
                title="Parallel Coordinates Plot for Feature Comparison", color_continuous_scale='bluered'
            )
            st.plotly_chart(fig)
            
            st.markdown("""
            <div class='graph-explanation'>
                <strong>Explanation:</strong> The parallel coordinates plot allows you to compare multiple features at once. 
                Each line represents a phone, and you can see how they compare across different specifications.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
            st.subheader("Radar Chart for Feature Comparison")
            radar_data = comparison_data[['Model', 'Memory', 'Storage', 'Rating', 'Selling Price', 'Points']]
            radar_data = radar_data.set_index('Model')
            fig = px.line_polar(radar_data, r='Memory', theta=radar_data.index, line_close=True, 
                               title="Radar Chart of Phone Features", color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig)
            
            st.markdown("""
            <div class='graph-explanation'>
                <strong>Explanation:</strong> The radar chart provides a quick visual comparison of key features. 
                Models with larger coverage area generally have better specifications.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
            st.subheader("Correlation Heatmap")
            corr = comparison_data[['Memory', 'Storage', 'Rating', 'Selling Price', 'Original Price', 'Points']].corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            st.pyplot(plt)
            
            st.markdown("""
            <div class='graph-explanation'>
                <strong>Explanation:</strong> The heatmap shows correlations between different features. 
                Values close to 1 indicate strong positive correlation, while values close to -1 show negative correlation.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("Please select at least 2 phones for comparison.")

    st.markdown("---")
    st.markdown("""
    <div class='footer'>
        © 2024 ProdRec Inc. | Built with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)

# Main App
def main():
    load_css_js()
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
    <style>
        .sidebar .sidebar-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .sidebar .sidebar-content .stSelectbox, .sidebar .sidebar-content .stMultiselect {
            color: #333;
        }
    </style>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox("Go to", ["Home", "Compare Products", "Compare Phones"], key="nav_select")
    
    if page == "Home":
        show_home()
    elif page == "Compare Products":
        show_compare()
    elif page == "Compare Phones":
        show_phone_comparison()

if __name__ == "__main__":
    main()
