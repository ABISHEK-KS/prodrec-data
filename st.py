import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Configurations
st.set_page_config(page_title="ProdRec - Product Comparison Platform",
                   page_icon="🛍️",
                   layout="wide")

# Helper Functions
@st.cache_data
def load_data(file_path):
    """Load dataset from a local CSV file and return a copy to avoid mutation."""
    return pd.read_csv(file_path).copy()

def parse_categories(data):
    """Parse categories with synonyms separated by '|'."""
    data['parsed_category'] = data['category'].apply(lambda x: x.split('|')[0])
    return data

def show_home():
    """Render the Home Page."""
    st.title("Welcome to **ProdRec**")
    st.subheader("Your Trusted Companion in Smarter Shopping Decisions")
    st.markdown("""
    At **ProdRec**, we simplify your shopping journey by helping you:
    - Discover products tailored to your needs and budget.
    - Compare features, prices, and ratings side-by-side.
    - Decide with confidence, backed by clear, transparent insights.

    Let’s make your shopping experience smarter, easier, and more rewarding.
    """)
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

    # Load Dataset
    DATA_FILE = "prodrecdata.csv"
    try:
        data = load_data(DATA_FILE)  # Now returns a copy, so it won't mutate cached data
        data = parse_categories(data)
    except Exception as e:
        st.error(f"Could not load the dataset. Make sure '{DATA_FILE}' exists in the same directory.\nError: {e}")
        return

    # Category Selection
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
        }).hide_index())

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

# Sidebar Button-style Navigation (CSS Customization)
sidebar_style = """
    <style>
    .sidebar .sidebar-content {
        background-color: #262730;
        color: white;
    }
    .sidebar .sidebar-content .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        width: 100%;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content .stButton>button:hover {
        background-color: #155a8a;
    }
    </style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)

# Button-based Sidebar Navigation with Session State
st.sidebar.title("Navigation")

# Initialize session state if not present
if 'page' not in st.session_state:
    st.session_state.page = 'home'  # Default page

# Sidebar buttons for page navigation
if st.sidebar.button("🏠 Home"):
    st.session_state.page = 'home'
elif st.sidebar.button("🔍 Compare Products"):
    st.session_state.page = 'compare'

# Show the selected page content
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'compare':
    show_compare()
