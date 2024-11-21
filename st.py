import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configurations
st.set_page_config(page_title="ProdRec - Product Comparison Platform", page_icon="🛍️", layout="wide")

# Load and Apply External CSS (if exists)
def load_css():
    """Reads external CSS and applies it."""
    try:
        with open("externalcss.css", "r") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file 'externalcss.css' not found. Make sure it is in the correct directory.")

# Helper Functions
@st.cache_data
def load_data(file_path):
    """Load dataset from a local Excel file and return a copy to avoid mutation."""
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
    st.markdown("""At **ProdRec**, we simplify your shopping journey by helping you:
    - Discover products tailored to your needs and budget.
    - Compare features, prices, and ratings side-by-side.
    - Decide with confidence, backed by clear, transparent insights.

    Let’s make your shopping experience smarter, easier, and more rewarding.""")
    st.image(
        "https://via.placeholder.com/1200x500.png?text=Your+Product+Comparison+Starts+Here!",
        use_column_width=True,
        caption="Explore. Compare. Decide."
    )
    st.markdown("---")
    st.markdown("### Ready to explore?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Explore Products"):
            st.success("Switch to 'Compare Products' using the sidebar.")
    with col2:
        if st.button("Learn More"):
            st.info("ProdRec makes smarter shopping possible.")
    st.markdown("---")
    st.markdown("© 2024 ProdRec Inc. | Built with ❤️ using Streamlit")

def show_compare():
    """Render the Product Comparison Page."""
    st.title("🔍 Compare Products")
    st.markdown("Compare products based on **category**, **price**, **ratings**, and other features.")

    # Load Dataset for General Comparison
    DATA_FILE = "pr.csv"
    try:
        data = load_data(DATA_FILE)  # Now returns a copy, so it won't mutate cached data
    except Exception as e:
        st.error(f"Could not load the dataset. Make sure '{DATA_FILE}' exists in the same directory.\nError: {e}")
        return

    # Category Selection for General Product Comparison
    unique_categories = data['parsed_category'].dropna().unique()
    selected_category = st.selectbox("Select a Category", unique_categories)

    # Filter Data
    filtered_data = data[data['parsed_category'] == selected_category]

    # Product Selection
    st.subheader("Select Products to Compare")
    product_options = filtered_data['product_name'].unique()
    selected_products = st.multiselect("Choose Products:", product_options, default=product_options[:2])

    if len(selected_products) >= 2:
        comparison_data = filtered_data[filtered_data['product_name'].isin(selected_products)]

        # Ensure numeric columns are treated as numeric
        comparison_data['discounted_price'] = pd.to_numeric(comparison_data['discounted_price'], errors='coerce')
        comparison_data['actual_price'] = pd.to_numeric(comparison_data['actual_price'], errors='coerce')
        comparison_data['rating'] = pd.to_numeric(comparison_data['rating'], errors='coerce')
        comparison_data['rating_count'] = pd.to_numeric(comparison_data['rating_count'], errors='coerce')

        # Display Comparison Table with Styling
        st.subheader("Comparison Table")
        table_data = comparison_data[['product_name', 'discounted_price', 'actual_price',
                                      'discount_percentage', 'rating', 'rating_count', 'about_product']]
        table_data = table_data.rename(columns={
            'product_name': 'Product Name',
            'discounted_price': 'Discounted Price',
            'actual_price': 'Actual Price',
            'discount_percentage': 'Discount (%)',
            'rating': 'Rating',
            'rating_count': 'Rating Count',
            'about_product': 'Description'
        })

        # Apply formatting only to numeric columns
        st.dataframe(table_data.style.format({
            'Discounted Price': '₹{:.2f}',
            'Actual Price': '₹{:.2f}',
            'Discount (%)': '{:.2f}%',
            'Rating': '{:.1f}',
            'Rating Count': '{:,}',
        }))

        # Visualizations

        # Price Bar Chart
        st.plotly_chart(px.bar(
            comparison_data, x='product_name', y=['discounted_price', 'actual_price'],
            barmode='group', title="Price Comparison",
            labels={"discounted_price": "Price (₹)", "actual_price": "Price (₹)"},
            color='product_name'
        ))

        # Ratings Bar Chart
        st.plotly_chart(px.bar(
            comparison_data, x='product_name', y='rating',
            title="Ratings Comparison", color='product_name',
            labels={"rating": "Rating (out of 5)"},
            color_discrete_map={'product_name': 'rgb(255, 165, 0)'}
        ))

        # Radar Chart
        st.subheader("Feature Comparison (Radar Chart)")
        radar_data = comparison_data[['product_name', 'discounted_price', 'rating', 'rating_count']].set_index('product_name')
        radar_data = radar_data.T  # Transpose for radar plotting
        radar_fig = go.Figure()
        for product in radar_data.columns:
            radar_fig.add_trace(go.Scatterpolar(
                r=radar_data[product],
                theta=radar_data.index,
                fill='toself',
                name=product
            ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title="Product Features Comparison"
        )
        st.plotly_chart(radar_fig)

        # Product Distribution Chart (Bubble Chart)
        st.subheader("Price vs Rating Distribution")
        fig = px.scatter(
            comparison_data, x='discounted_price', y='rating', size='rating_count', hover_name='product_name',
            title="Price vs Rating", labels={'discounted_price': 'Price (₹)', 'rating': 'Rating'},
            color='rating', size_max=60, color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig)

    else:
        st.warning("Please select at least 2 products for comparison.")

    st.markdown("---")
    st.markdown("© 2024 ProdRec Inc. | Built with ❤️ using Streamlit")

# **Phone Comparison Functionality** (New Section)

# Load the phone data
phone_data = pd.read_csv('prp.csv')

def show_phone_comparison():
    """Render Phone Comparison Section with Enhanced Visuals."""
    st.title("📱 Compare Phones")

    # Step 1: Select Phones for Comparison (Multiple Brands)
    brands = phone_data['Brand'].unique()
    selected_brands = st.multiselect("Select Brands", brands, default=brands[:1])

    # Prepare list to store selected phones
    selected_phones = []

    for brand in selected_brands:
        brand_data = phone_data[phone_data['Brand'] == brand]
        models = brand_data['Model'].unique()
        selected_model = st.selectbox(f"Select Phone from {brand}", models)
        selected_phones.append(selected_model)

    if len(selected_phones) >= 2:
        # Filter the data for selected models
        comparison_data = phone_data[phone_data['Model'].isin(selected_phones)]

        # Display comparison table
        st.subheader("Phone Comparison Table")
        table_data = comparison_data[['Brand', 'Model', 'Color', 'Memory', 'Storage', 'Rating', 'Selling Price', 'Original Price']]
        table_data = table_data.rename(columns={
            'Brand': 'Brand',
            'Model': 'Model',
            'Color': 'Color',
            'Memory': 'Memory',
            'Storage': 'Storage',
            'Rating': 'Rating',
            'Selling Price': 'Selling Price (₹)',
            'Original Price': 'Original Price (₹)'
        })
        st.dataframe(table_data)

        # **1. Price Distribution Plot**
        st.subheader("Price Distribution of Selected Phones")
        fig_price_dist = px.box(
            comparison_data, x='Model', y='Selling Price',
            title="Price Distribution for Selected Phones",
            labels={'Selling Price': 'Price (₹)', 'Model': 'Phone Model'},
            points='all'
        )
        st.plotly_chart(fig_price_dist)

        # **2. Storage vs. Memory Scatter Plot**
        st.subheader("Storage vs Memory Scatter Plot")
        fig_storage_memory = px.scatter(
            comparison_data, 
            x='Storage', 
            y='Memory', 
            color='Model', 
            size='Selling Price', 
            hover_name='Model', 
            title="Storage vs Memory for Selected Phones",
            labels={'Storage': 'Storage (GB)', 'Memory': 'Memory (GB)'}
        )
        st.plotly_chart(fig_storage_memory)

        # **3. Rating Distribution Histogram**
        st.subheader("Rating Distribution Histogram")
        fig_rating_dist = px.histogram(
            comparison_data, 
            x='Rating', 
            color='Model', 
            title="Rating Distribution for Selected Phones",
            labels={'Rating': 'Rating (out of 5)'}
        )
        fig_rating_dist.update_layout(bargap=0.2)
        st.plotly_chart(fig_rating_dist)

        # **4. Comparison of Colors (Pie Chart)**
        st.subheader("Phone Colors Distribution")
        color_counts = comparison_data['Color'].value_counts()
        fig_color_dist = px.pie(
            names=color_counts.index, 
            values=color_counts.values, 
            title="Color Distribution of Selected Phones"
        )
        st.plotly_chart(fig_color_dist)

        # **Price Bar Chart for Phones**
        st.plotly_chart(px.bar(
            comparison_data, x='Model', y=['Selling Price', 'Original Price'],
            barmode='group', title="Price Comparison of Selected Phones",
            labels={"Selling Price": "Selling Price (₹)", "Original Price": "Original Price (₹)"}
        ))

    else:
        st.warning("Please select at least 2 phones for comparison.")
    
# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Compare Products", "Compare Phones"])

if page == "Home":
    load_css()
    show_home()

elif page == "Compare Products":
    load_css()
    show_compare()

elif page == "Compare Phones":
    load_css()
    show_phone_comparison()
