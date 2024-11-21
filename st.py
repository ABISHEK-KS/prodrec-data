import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

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
        st.subheader("Price Comparison")
        st.plotly_chart(px.bar(
            comparison_data, x='product_name', y=['discounted_price', 'actual_price'],
            barmode='group', title="Price Comparison",
            labels={"discounted_price": "Price (₹)", "actual_price": "Price (₹)"},
            color='product_name'
        ))

        # Ratings Bar Chart
        st.subheader("Rating Comparison")
        st.plotly_chart(px.bar(
            comparison_data, x='product_name', y='rating',
            title="Ratings Comparison", color='product_name',
            labels={"rating": "Rating (out of 5)"},
            color_discrete_map={'product_name': 'rgb(255, 165, 0)'}
        ))

        # Radar Chart
        st.subheader("Features Comparison")
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
        st.subheader("Price vs Rating Comparison")
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
    """Render Phone Comparison Section with Enhanced Visuals and Recommendations."""
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

        # **Convert 'Memory' and 'Storage' to numeric (cleaning data properly)**
        comparison_data['Memory'] = comparison_data['Memory'].str.replace('GB', '').astype(int)
        comparison_data['Storage'] = comparison_data['Storage'].str.replace('GB', '').astype(int)

        # Assign points for each feature
        def assign_points(phone):
            points = 0

            # Points for Memory (RAM) (1 point per GB of RAM)
            points += phone['Memory']

            # Points for Storage (1 point per 64 GB of Storage)
            points += phone['Storage'] / 64  # Divide by 64 to scale it reasonably

            # Points for Rating (Maximum 10 points for a 5-star rating)
            points += phone['Rating'] * 2  # 2 points per rating star

            # Points for Selling Price (Inverse scoring: lower price gets more points)
            points += 100000 / phone['Selling Price']  # Arbitrary scaling to make it comparable

            return points

        # Calculate points for each phone
        comparison_data['Points'] = comparison_data.apply(assign_points, axis=1)

        # Display comparison table
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
        st.dataframe(table_data)

        # **Phone Recommendation: Best Overall Score**
        st.markdown("### 📱 Recommended Phone: Best Features")
        # Find the phone with the highest total points
        best_phone = comparison_data.loc[comparison_data['Points'].idxmax()]
        st.write(f"The phone with the **best features** is **{best_phone['Model']}** by {best_phone['Brand']}.")
        st.write(f"**Total Points**: {best_phone['Points']}")
        st.write(f"**Selling Price**: ₹{best_phone['Selling Price']}")
        st.write(f"**Rating**: {best_phone['Rating']} stars")

        # 1. **Price vs Rating Scatter Plot**
        st.subheader("Price vs Rating")
        st.plotly_chart(px.scatter(
            comparison_data, x='Selling Price', y='Rating',
            title="Price vs Rating Comparison", color='Model',
            labels={'Selling Price': 'Price (₹)', 'Rating': 'Rating (out of 5)'}
        ))

        # 2. **Memory Distribution Histogram**
        st.subheader("Memory (RAM) Distribution")
        fig = plt.figure()
        sns.histplot(comparison_data['Memory'], kde=True, color='skyblue')
        plt.title('Distribution of Memory (RAM) in GB')
        st.pyplot(fig)

        # 3. **Storage Distribution Histogram**
        st.subheader("Storage Distribution")
        fig = plt.figure()
        sns.histplot(comparison_data['Storage'], kde=True, color='green')
        plt.title('Distribution of Storage (GB)')
        st.pyplot(fig)

        # 4. **Rating Distribution Histogram**
        st.subheader("Rating Distribution")
        fig = plt.figure()
        sns.histplot(comparison_data['Rating'], kde=True, color='orange')
        plt.title('Distribution of Ratings')
        st.pyplot(fig)

        # 5. **Price Distribution Box Plot**
        st.subheader("Price Distribution (Box Plot)")
        fig = plt.figure()
        sns.boxplot(x=comparison_data['Selling Price'], color='lightcoral')
        plt.title('Price Distribution')
        st.pyplot(fig)

        # 6. **RAM vs Storage Scatter Plot**
        st.subheader("RAM vs Storage")
        st.plotly_chart(px.scatter(
            comparison_data, x='Memory', y='Storage',
            title="RAM vs Storage", color='Model',
            labels={'Memory': 'Memory (GB)', 'Storage': 'Storage (GB)'}
        ))

        # 7. **Price vs Memory Scatter Plot**
        st.subheader("Price vs Memory")
        st.plotly_chart(px.scatter(
            comparison_data, x='Selling Price', y='Memory',
            title="Price vs Memory (RAM)", color='Model',
            labels={'Selling Price': 'Price (₹)', 'Memory': 'Memory (GB)'}
        ))

        # 8. **Heatmap of Feature Correlations**
        st.subheader("Feature Correlation Heatmap")
        corr_matrix = comparison_data[['Memory', 'Storage', 'Rating', 'Selling Price', 'Points']].corr()
        fig = plt.figure()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        st.pyplot(fig)

        # 9. **Feature Comparison Radar Chart (Spider Plot)**
        st.subheader("Features Comparison (Radar Chart)")
        radar_data = comparison_data[['Model', 'Memory', 'Storage', 'Rating', 'Selling Price']].set_index('Model')
        radar_data = radar_data.T  # Transpose for radar plotting
        radar_fig = go.Figure()
        for model in radar_data.columns:
            radar_fig.add_trace(go.Scatterpolar(
                r=radar_data[model],
                theta=radar_data.index,
                fill='toself',
                name=model
            ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title="Phone Features Comparison"
        )
        st.plotly_chart(radar_fig)

        # 10. **Feature Rating Breakdown Table**
        st.subheader("Feature Rating Breakdown")
        rating_table = comparison_data[['Model', 'Memory', 'Storage', 'Rating', 'Selling Price', 'Points']]
        rating_table['Memory Points'] = rating_table['Memory']
        rating_table['Storage Points'] = rating_table['Storage'] / 64
        rating_table['Rating Points'] = rating_table['Rating'] * 2
        rating_table['Price Points'] = 100000 / rating_table['Selling Price']
        
        st.write(rating_table)

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
