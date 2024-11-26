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

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def load_data(file_path):
    """Load dataset from a local CSV file and return a copy to avoid mutation."""
    data = pd.read_csv(file_path)
    
    # Check if 'Category' column exists (case-sensitive check)
    if 'Category' not in data.columns:
        raise ValueError("'Category' column not found in the dataset.")
    
    # Parse categories (handle cases where data might have multiple categories or issues)
    data = parse_categories(data)
    
    # Clean other columns such as price, ratings
    data['discounted_price'] = pd.to_numeric(data['Price now'], errors='coerce')
    data['actual_price'] = pd.to_numeric(data['Earlier Price'], errors='coerce')
    data['rating'] = pd.to_numeric(data['Rating'], errors='coerce')
    data['rating_count'] = pd.to_numeric(data['Rating Count'], errors='coerce')

    return data

def parse_categories(data):
    """Parse categories with synonyms separated by '|'."""
    data['parsed_category'] = data['Category'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else 'Unknown')
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

@st.cache_data
def load_data(file_path):
    """Load dataset from a local CSV file and return a copy to avoid mutation."""
    data = pd.read_csv(file_path)
    
    # Check if 'Category' column exists (case-sensitive check)
    if 'Category' not in data.columns:
        raise ValueError("'Category' column not found in the dataset.")
    
    # Parse categories (handle cases where data might have multiple categories or issues)
    data = parse_categories(data)
    
    # Clean other columns such as price, ratings
    data['discounted_price'] = pd.to_numeric(data['Price now'], errors='coerce')
    data['actual_price'] = pd.to_numeric(data['Earlier Price'], errors='coerce')
    data['rating'] = pd.to_numeric(data['Rating'], errors='coerce')
    data['rating_count'] = pd.to_numeric(data['Rating Count'], errors='coerce')

    return data

def parse_categories(data):
    """Parse categories if they contain multiple values separated by '|'. For now, we take the first category."""
    data['parsed_category'] = data['Category'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else 'Unknown')
    return data

@st.cache_data
def load_data(file_path):
    """Load dataset from a local CSV file and return a copy to avoid mutation."""
    data = pd.read_csv(file_path)
    
    # Check if 'Category' column exists (case-sensitive check)
    if 'Category' not in data.columns:
        raise ValueError("'Category' column not found in the dataset.")
    
    # Parse categories (handle cases where data might have multiple categories or issues)
    data = parse_categories(data)
    
    # Clean other columns such as price, ratings
    data['discounted_price'] = pd.to_numeric(data['Price now'], errors='coerce')
    data['actual_price'] = pd.to_numeric(data['Earlier Price'], errors='coerce')
    data['rating'] = pd.to_numeric(data['Rating'], errors='coerce')
    data['rating_count'] = pd.to_numeric(data['Rating Count'], errors='coerce')

    return data

def parse_categories(data):
    """Parse categories if they contain multiple values separated by '|'. For now, we take the first category."""
    data['parsed_category'] = data['Category'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else 'Unknown')
    return data

@st.cache_data
def load_data(file_path):
    """Load dataset from a local CSV file and return a copy to avoid mutation."""
    data = pd.read_csv(file_path)
    
    # Check if 'Category' column exists (case-sensitive check)
    if 'Category' not in data.columns:
        raise ValueError("'Category' column not found in the dataset.")
    
    # Parse categories (handle cases where data might have multiple categories or issues)
    data = parse_categories(data)
    
    # Clean other columns such as price, ratings
    data['discounted_price'] = pd.to_numeric(data['Price now'], errors='coerce')
    data['actual_price'] = pd.to_numeric(data['Earlier Price'], errors='coerce')
    data['rating'] = pd.to_numeric(data['Rating'], errors='coerce')
    data['rating_count'] = pd.to_numeric(data['Rating Count'], errors='coerce')

    return data

def parse_categories(data):
    """Parse categories if they contain multiple values separated by '|'. For now, we take the first category."""
    data['parsed_category'] = data['Category'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else 'Unknown')
    return data

def show_compare():
    """Render the Product Comparison Page with Detailed Descriptions and Point System."""
    st.title("🔍 Advanced Product Comparison")
    st.markdown("Compare products based on **category**, **price**, **ratings**, and other features.")

    # Load Dataset for General Comparison
    DATA_FILE = "opd.csv"  # Use your actual file path
    data = load_data(DATA_FILE)

    # Category Selection for General Product Comparison
    unique_categories = data['parsed_category'].dropna().unique()
    selected_category = st.selectbox("Select a Category", unique_categories)

    # Filter Data based on Selected Category
    filtered_data = data[data['parsed_category'] == selected_category]

    # Product Selection
    st.subheader("Select Products to Compare")
    product_options = filtered_data['Name'].unique()  # Use 'Name' for product names
    selected_products = st.multiselect("Choose Products:", product_options, default=product_options[:2])

    if len(selected_products) >= 2:
        comparison_data = filtered_data[filtered_data['Name'].isin(selected_products)]

        # Detailed Description and Image for each product
        for product in selected_products:
            st.subheader(f"Product Description: {product}")
            product_data = comparison_data[comparison_data['Name'] == product].iloc[0]
            
            # Display product image from 'Media' column (not 'Image')
            image_url = product_data['Media']
            st.image(image_url, caption=f"{product} Image", use_column_width=True)

            # Product Description
            st.markdown(f"**Category**: {product_data['parsed_category']}")
            st.markdown(f"**Discounted Price**: ₹{product_data['discounted_price']:.2f}")
            st.markdown(f"**Actual Price**: ₹{product_data['actual_price']:.2f}")
            st.markdown(f"**Rating**: {product_data['rating']}/5")
            st.markdown(f"**Rating Count**: {product_data['rating_count']}")

            # Show top reviews (you can customize this section as needed)
            st.markdown(f"### Reviews for {product}")
            st.markdown("#### Top Reviews:")
            reviews = product_data['Reviews'].split(",")  # Assuming reviews are comma-separated
            st.markdown(f"1. **{reviews[0]}**")
            st.markdown(f"2. **{reviews[1]}**")

            st.markdown("---")

        # Visualizations (Radar chart, Heatmap, etc.)
        st.subheader("Product Features Comparison (Radar Chart)")
        radar_data = comparison_data[['Name', 'discounted_price', 'rating', 'rating_count']].set_index('Name')
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

        # Heatmap of Correlations
        st.subheader("Correlation Heatmap")
        corr_matrix = comparison_data[['discounted_price', 'actual_price', 'rating', 'rating_count']].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

        # Assign Points Based on Features (Example: Higher rating gets more points)
        st.subheader("Point-Based Comparison Table")
        points_data = comparison_data[['Name', 'rating', 'discounted_price', 'rating_count']].copy()
        points_data['rating_points'] = points_data['rating'] * 10  # Example scoring logic: 10 points per rating point
        points_data['price_points'] = points_data['discounted_price'].apply(lambda x: 1000 - x)  # Example: cheaper products get more points
        points_data['total_points'] = points_data['rating_points'] + points_data['price_points']
        points_data = points_data[['Name', 'rating_points', 'price_points', 'total_points']]

        st.dataframe(points_data)

        # Recommendation: Product with the highest points
        best_product = points_data.loc[points_data['total_points'].idxmax()]
        st.subheader("Recommended Product")
        st.markdown(f"The product with the most points is **{best_product['Name']}** with **{best_product['total_points']} points**!")

    else:
        st.warning("Please select at least 2 products for comparison.")

    st.markdown("---")
    st.markdown("© 2024 ProdRec Inc. | Built with ❤️ using Streamlit")







    
# Load the phone data
phone_data = pd.read_csv('prp.csv')

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Load the phone data
phone_data = pd.read_csv('prp.csv')

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def show_phone_comparison():
    """Render Phone Comparison Section with Advanced and Aesthetically Pleasing Visuals."""
    st.title("📱 Compare Phones")

    # Step 1: Select Phones for Comparison (Multiple Brands)
    brands = phone_data['Brand'].unique()
    selected_brands = st.multiselect("Select Brands", brands, default=brands[:1], key="brand_selector")

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

        # Convert 'Memory' and 'Storage' to numeric (removing 'GB' and converting to integer)
        comparison_data['Memory'] = comparison_data['Memory'].str.replace('GB', '').astype(int)
        comparison_data['Storage'] = comparison_data['Storage'].str.replace('GB', '').astype(int)

        # Assign points for each feature (optional, based on your logic)
        def assign_points(phone):
            points = 0
            points += phone['Memory']  # Points for Memory (1 point per GB of RAM)
            points += phone['Storage'] / 64  # Points for Storage (1 point per 64 GB of Storage)
            points += phone['Rating'] * 2  # Points for Rating (Maximum 10 points for a 5-star rating)
            points += 100000 / phone['Selling Price']  # Inverse scoring: lower price gets more points
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
        best_phone = comparison_data.loc[comparison_data['Points'].idxmax()]
        st.markdown("### 📱 **Recommended Phone with the Best Features**")
        st.write(f"The phone with the **best features** is **{best_phone['Model']}** by {best_phone['Brand']}.")
        st.write(f"**Total Points**: {best_phone['Points']}")
        st.write(f"**Selling Price**: ₹{best_phone['Selling Price']}")
        st.write(f"**Rating**: {best_phone['Rating']} stars")
        st.write(f"**Best Buy**: This phone offers the most balanced combination of **features, performance**, and **price**!")

        # Visualizations

        # **Visual 1: Violin Plot of Rating Distribution by Model**
        st.subheader("Rating Distribution by Model (Violin Plot)")
        fig = px.violin(comparison_data, y='Rating', x='Model', color='Model', box=True, points="all", title="Rating Distribution by Model")
        st.plotly_chart(fig)

        # **Visual 2: Violin Plot of Selling Price Distribution by Model**
        st.subheader("Selling Price Distribution by Model (Violin Plot)")
        fig = px.violin(comparison_data, y='Selling Price', x='Model', color='Model', box=True, points="all", title="Selling Price Distribution by Model")
        st.plotly_chart(fig)

        # **Visual 3: Violin Plot of Memory (RAM) Distribution by Model**
        st.subheader("Memory (RAM) Distribution by Model (Violin Plot)")
        fig = px.violin(comparison_data, y='Memory', x='Model', color='Model', box=True, points="all", title="Memory (RAM) Distribution by Model")
        st.plotly_chart(fig)

        # **Visual 4: Parallel Coordinates Plot (for feature comparison)**
        st.subheader("Parallel Coordinates Plot")
        fig = px.parallel_coordinates(
            comparison_data, color="Points", dimensions=["Memory", "Storage", "Rating", "Selling Price", "Points"],
            title="Parallel Coordinates Plot for Feature Comparison"
        )
        st.plotly_chart(fig)

        # **Visual 5: Radar Chart for Feature Comparison**
        st.subheader("Radar Chart for Feature Comparison")
        radar_data = comparison_data[['Model', 'Memory', 'Storage', 'Rating', 'Selling Price', 'Points']]
        radar_data = radar_data.set_index('Model')
        fig = px.line_polar(radar_data, r='Memory', theta=radar_data.index, line_close=True, title="Radar Chart of Phone Features")
        st.plotly_chart(fig)

        # **Visual 6: Correlation Heatmap (Numerical Data)**
        st.subheader("Correlation Heatmap")
        corr = comparison_data[['Memory', 'Storage', 'Rating', 'Selling Price', 'Original Price', 'Points']].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        st.pyplot(plt)

        # **Visual 7: Distribution of Points (Histogram)**
        st.subheader("Points Distribution")
        fig = px.histogram(comparison_data, x='Points', title="Distribution of Points across Phones", color='Model')
        st.plotly_chart(fig)

        # **Visual 8: Line Plot of Selling Price vs Rating (Trend analysis)**
        st.subheader("Selling Price vs Rating Trend (Line Plot)")
        fig = px.line(comparison_data, x='Selling Price', y='Rating', color='Model', markers=True, title="Selling Price vs Rating Trend")
        st.plotly_chart(fig)

        # **Visual 9: Line Plot of Storage vs Points (Trend analysis)**
        st.subheader("Storage vs Points Trend (Line Plot)")
        fig = px.line(comparison_data, x='Storage', y='Points', color='Model', markers=True, title="Storage vs Points Trend")
        st.plotly_chart(fig)

        # **Visual 10: Line Plot of Memory vs Selling Price (Trend analysis)**
        st.subheader("Memory (RAM) vs Selling Price Trend (Line Plot)")
        fig = px.line(comparison_data, x='Memory', y='Selling Price', color='Model', markers=True, title="Memory (RAM) vs Selling Price Trend")
        st.plotly_chart(fig)

# Call the function to show phone comparison when running the app







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
