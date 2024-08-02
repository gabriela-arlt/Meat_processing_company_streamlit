
import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
import numpy as np
import plotly.graph_objs as go
import base64
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go


# Set Streamlit layout to wide
st.set_page_config(layout="wide")


####### ANIMATION ############################################    
    # Function to encode image to base64

# Function to encode image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Define paths to your local images
image_paths = ["bireni.png", "dobrudja.png", "nadenitsa.png", "burgas.png", "bireni.png", "dobrudja.png", "nadenitsa.png", "burgas.png", "bireni.png", "dobrudja.png", "nadenitsa.png", "burgas.png", "bireni.png", "dobrudja.png", "nadenitsa.png", "burgas.png", "bireni.png", "dobrudja.png", "nadenitsa.png", "burgas.png", "bireni.png", "dobrudja.png", "nadenitsa.png", "burgas.png", "bireni.png", "dobrudja.png", "nadenitsa.png"]

# Encode images to base64
encoded_images = [image_to_base64(image) for image in image_paths]

# HTML and JavaScript for Falling Images
falling_images_html = """
<!DOCTYPE html>
<html>
<head>
<style>
@keyframes fall {{
  0% {{ top: -10%; opacity: 1; }}
  100% {{ top: 110%; opacity: 0; }}
}}

.falling {{
  position: fixed;
  top: -10%;
  width: 100px;  /* Image width */
  height: 100px; /* Image height */
  animation: fall {0}s linear infinite;
  opacity: 0;
}}

#container {{
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 9999;
  pointer-events: none;
}}

.falling img {{
  width: 100px;  /* Image width */
  height: 100px; /* Image height */
}}

</style>
</head>
<body>

<div id="container">
  {1}
</div>

<script>
setTimeout(function() {{
  document.getElementById('container').style.display = 'none';
}}, 5000);
</script>

</body>
</html>
"""

# Generate image divs
image_divs = ""
for i in range(30):
    right_position = np.random.randint(0, 100)
    left_position = np.random.randint(0, 100)
    animation_delay = np.random.uniform(0, 5)  # Increase range for more randomness
    animation_duration = np.random.uniform(5, 10)  # Increase duration for longer fall
    image_divs += f'<div class="falling" style="left: {right_position}%; animation-delay: {animation_delay}s; animation-duration: {animation_duration}s;"><img src="data:image/png;base64,{encoded_images[i % len(encoded_images)]}" alt="image{i}"></div>'
    image_divs += f'<div class="falling" style="left: {left_position}%; animation-delay: {animation_delay}s; animation-duration: {animation_duration}s;"><img src="data:image/png;base64,{encoded_images[i % len(encoded_images)]}" alt="image{i}"></div>'

# Format HTML with animation duration and generated image divs
falling_images_html = falling_images_html.format(70, image_divs)  # Using 10 seconds as an example duration

# Main Streamlit app code
# Your existing dashboard code goes here...

# Add checkbox at the bottom
st.markdown("---")  # Optional: Add a horizontal line for separation
if st.checkbox("Surprise...surprise..."):
    st.components.v1.html(falling_images_html, height=600)


# Center the logo image
#col_center = st.columns([1, 2, 1])[1]
#with col_center:
 #   st.image("UNICAL_1.png", width=600)
# Create columns with specified relative widths
col_left, col_center, col_right = st.columns([1, 2, 1])

# Display images in each column with inline styling
with col_left:
    st.image("nadenitsa.png", use_column_width=True)

with col_center:
    st.image("Executive_dashboard_unical.png", use_column_width=True)

with col_right:
    st.image("burgas.png", use_column_width=True)

# Load CSS from file
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
load_css("styles.css")

# Main data
@st.cache_data
def load_data_from_url():
    # Google Sheets URL with export format
    sheet_url = "https://docs.google.com/spreadsheets/d/1utFaEaOylXT7fFuXUWZgFwQ-tKBewG1KRbXHoerdtzs/export?format=csv"

    # Specify data types for columns to avoid mixed type issues
    dtype = {
        'Customer_country': str,
        'Sales': float,
        'Document_number': str,
        'Latitude': float,
        'Longitude': float,
        'Partner_code': str,
        'Product': str,
        'Order_date': str,  # assuming date is read as string
        'Quantity': float,
        'Unit_price': float,
        'City': str,  # Add this line
        'Distribution_type': str,  # Add this line
        'Quantity_measurement': str  # Add this line
    }

    # Read the data into a DataFrame
    df = pd.read_csv(sheet_url, dtype=dtype, low_memory=False)
    return df

data = load_data_from_url()

# Dropdown for the dataframe
with st.expander("Data Preview"):
    st.dataframe(data)

# Convert date column to datetime format
data['Order_date'] = pd.to_datetime(data['Order_date'])
data_kpi = data.copy()
data_second = data.copy()

# Define the date range for the slider
min_date = data['Order_date'].min()
max_date = data['Order_date'].max()

# Create a list of monthly periods
monthly_periods = pd.date_range(start=min_date, end=max_date, freq='M').to_period('M')

# Add a slider for selecting the date range
start_period, end_period = st.select_slider(
    "Select date range:",
    options=monthly_periods,
    value=(monthly_periods[0], monthly_periods[-1])
)

# Add multiselect filters in the sidebar
with st.sidebar:
    # Country filter with "All" option
    countries = data['Customer_country'].unique()
    countries = np.insert(countries, 0, "All")
    selected_country = st.selectbox(
        "Select Country:",
        options=countries
    )

    # Filter the cities based on selected country or show all cities if "All" is selected
    if selected_country == "All":
        filtered_cities = data['City'].unique()
    else:
        filtered_cities = data[data['Customer_country'] == selected_country]['City'].unique()

    selected_cities = st.multiselect(
        "Select City:",
        options=filtered_cities,
        default=filtered_cities  # Select all options by default
    )

    # Distribution type filter
    distribution_types = data['Distribution_type'].unique()
    selected_distribution_types = st.multiselect(
        "Select Distribution Type:",
        options=distribution_types,
        default=distribution_types  # Select all options by default
    )

# Filter data based on selected filters
if selected_country == "All":
    filtered_data = data[
        (data['City'].isin(selected_cities)) &
        (data['Distribution_type'].isin(selected_distribution_types))
    ].copy()
else:
    filtered_data = data[
        (data['Customer_country'] == selected_country) &
        (data['City'].isin(selected_cities)) &
        (data['Distribution_type'].isin(selected_distribution_types))
    ].copy()

# Data preprocessing for first row
total_sales_per_product_type = filtered_data.groupby('Product_type')['Sales'].sum().reset_index()
total_orders_per_product_type = filtered_data.groupby('Product_type')['Document_number'].nunique().reset_index()
average_order_value_per_product_type = filtered_data.groupby('Product_type')['Sales'].mean().reset_index()


# Your existing code to create charts using the generated colors

# Function to update trace textinfo conditionally
def update_traces_conditional(fig, show_top_n=1):
    for trace in fig.data:
        values = trace['values']
        labels = trace['labels']
        top_values_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:show_top_n]
        trace.text = [
            f"{label}<br>{value}" if i in top_values_indices else ''
            for i, (label, value) in enumerate(zip(labels, values))
        ]
        trace.hovertext = [
            f"{label}<br>{value}"
            for label, value in zip(labels, values)
        ]
        trace.textinfo = 'text'
        trace.hoverinfo = 'text'

# Function to split long labels into multiple lines
def split_labels(labels, max_length=15):
    def split_label(label):
        words = label.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > max_length:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word) + 1  # Account for space
            else:
                current_line.append(word)
                current_length += len(word) + 1
        
        lines.append(' '.join(current_line))  # Append remaining words
        return '<br>'.join(lines)

    return [split_label(label) for label in labels]

# Apply split_labels function to product types
total_sales_per_product_type['Product_type'] = split_labels(total_sales_per_product_type['Product_type'])
average_order_value_per_product_type['Product_type'] = split_labels(average_order_value_per_product_type['Product_type'])
total_orders_per_product_type['Product_type'] = split_labels(total_orders_per_product_type['Product_type'])

# Function to create pie charts with the largest slice exploded
def create_pie_chart_with_exploded_slice(data, values_column, names_column, title):
    # Check if the data is empty
    if data.empty:
        st.warning(f"No data available for {title}. Please adjust the filters.")
        return None
    
    largest_slice_index = data[values_column].idxmax()
    pull_values = [0.1 if i == largest_slice_index else 0 for i in range(len(data))]
    
    # Create the pie chart with automatic colors
    pie_chart = go.Pie(
        labels=data[names_column],
        values=data[values_column],
        hole=0.3,
        pull=pull_values,
        textposition='inside'
    )
    
    # Create the figure
    fig = go.Figure(data=[pie_chart])
    fig.update_layout(
        title=dict(text=title, x=0.25),
        paper_bgcolor='#FFDAB9',
        margin=dict(l=38, r=38, t=58, b=38),
        showlegend=False
    )
    
    return fig

# Total Sales per Product Type
fig1 = create_pie_chart_with_exploded_slice(total_sales_per_product_type, 'Sales', 'Product_type', 'Total Sales per Product Type')
if fig1:
    update_traces_conditional(fig1)

# Average Order Value per Product Type
fig2 = create_pie_chart_with_exploded_slice(average_order_value_per_product_type, 'Sales', 'Product_type', 'Average Order Value per Product Type')
if fig2:
    update_traces_conditional(fig2)

# Total Orders per Product Type
fig3 = create_pie_chart_with_exploded_slice(total_orders_per_product_type, 'Document_number', 'Product_type', 'Total Orders per Product Type')
if fig3:
    update_traces_conditional(fig3)

# Row 1 with 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    if fig1:
        st.plotly_chart(fig1, use_container_width=True)
with col2:
    if fig3:
        st.plotly_chart(fig3, use_container_width=True)
with col3:
    if fig2:
        st.plotly_chart(fig2, use_container_width=True)

# Data Processing Row 2

# Assuming 'Order_date' is a datetime column
filtered_data['Month_Year'] = filtered_data['Order_date'].dt.to_period('M')

# Group data by month and year, calculate total sales per month
monthly_sales_month_year = filtered_data.groupby('Month_Year')['Sales'].sum().reset_index()

# Convert 'Month_Year' back to datetime for Plotly
monthly_sales_month_year['Month_Year'] = monthly_sales_month_year['Month_Year'].dt.to_timestamp()

# Plotting with Plotly
fig4 = px.line(monthly_sales_month_year, x='Month_Year', y='Sales', title='Monthly Sales')

# Customize line properties to change the color to red
#fig4.update_traces(line=dict(width=3, color='red'))

# Create the shadow line
shadow_line = go.Scatter(
    x=monthly_sales_month_year['Month_Year'],
    y=monthly_sales_month_year['Sales'],
    mode='lines',
    line=dict(color='rgba(0, 0, 0, 0.3)', width=12),
    showlegend=False
)

# Create the main line
main_line = go.Scatter(
    x=monthly_sales_month_year['Month_Year'],
    y=monthly_sales_month_year['Sales'],
    mode='lines',
    line=dict(color='red', width=4),
    name='Monthly Sales'
)

# Combine both lines in the figure
fig4 = go.Figure(data=[shadow_line, main_line])

# Update layout for X-axis labels and responsiveness, and remove grid lines
fig4.update_layout(
    title=dict(text='Monthly Sales', x=0.5),
    xaxis=dict(
        tickmode='linear',
        dtick='M1',
        tickformat="%b %Y",
        showgrid=False  # Remove x-axis grid lines
    ),
    yaxis=dict(
        showgrid=False  # Remove y-axis grid lines
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='BlanchedAlmond',
    margin=dict(l=10, r=10, t=20, b=10),
    width=1000  # Adjust width as needed for your layout
)

# Plotting Yearly Growth

# Group by year and sum the sales
data['Year'] = data['Order_date'].dt.year
annual_sales = data.groupby('Year')['Sales'].sum().reset_index()

# Calculate YoY growth
annual_sales['YoY_Growth'] = annual_sales['Sales'].pct_change() * 100

# Create the bar chart using Plotly Express
fig_g = px.bar(annual_sales, x='Year', y='Sales', text='Sales',
               title='Year-on-Year Sales Growth', labels={'Sales': 'Total Sales'}, color_discrete_sequence=['#800000'])

# Add YoY Growth annotation
for i in range(1, len(annual_sales)):
    fig_g.add_annotation(x=annual_sales['Year'][i], y=annual_sales['Sales'][i],
                         text=f"{annual_sales['YoY_Growth'][i]:.2f}% YoY",
                         showarrow=False, yshift=10, font=dict(color="red", size=12))

# Update layout
fig_g.update_layout(
    xaxis_title='2022           2023',
    yaxis_title='Sales',
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    yaxis=dict(showticklabels=False),
    xaxis=dict(showticklabels=False)
)

# Display the updated chart (assuming you have a column for displaying the chart)
row2_col1, row2_col2, row2_col3 = st.columns([3, 7, 1])
with row2_col2:
    st.plotly_chart(fig4, use_container_width=True)
with row2_col1:
    st.plotly_chart(fig_g, use_container_width=True)


# Mapping

data = filtered_data

# Aggregate sales by latitude and longitude
aggregated_data = data.groupby(['Latitude', 'Longitude'], as_index=False).agg({'Sales': 'sum'})

# Extract relevant columns
latitude = aggregated_data['Latitude']
longitude = aggregated_data['Longitude']
sales = aggregated_data['Sales']

# Create a folium map
sales_map = folium.Map(location=[latitude.mean(), longitude.mean()], zoom_start=7)

# Add circle markers to the map
for lat, lon, sale in zip(latitude, longitude, sales):
    folium.CircleMarker(
        location=[lat, lon],
        radius=sale / 150000,  # Adjust the radius based on sales amount for better visualization
        color='red',
        fill=True,
        fill_color='red',
        popup=f'Sales: ${sale:,.2f}'  # Display the sum of sales in the popup
    ).add_to(sales_map)

# Data Preparation

# Extract the year from the Order_date
data['Year'] = data['Order_date'].dt.year

# Calculate number of unique clients per year
clients_per_year = data.groupby('Year')['Partner_code'].nunique().reset_index()

# Calculate repeat purchase rate per year
repeat_clients = data.groupby(['Year', 'Partner_code']).size().reset_index(name='counts')
repeat_clients = repeat_clients[repeat_clients['counts'] > 1]
repeat_clients_per_year = repeat_clients.groupby('Year')['Partner_code'].nunique().reset_index()
total_clients_per_year = data.groupby('Year')['Partner_code'].nunique().reset_index()
repeat_clients_per_year = pd.merge(repeat_clients_per_year, total_clients_per_year, on='Year', suffixes=('_repeat', '_total'))
repeat_clients_per_year['Repeat_rate'] = repeat_clients_per_year['Partner_code_repeat'] / repeat_clients_per_year['Partner_code_total']

# Plotting

# Define the colors for the years
colors = ['#FA8072' if year % 2 == 0 else '#800000' for year in clients_per_year['Year']]

# Create the number of clients per year bar chart
clients_bar_chart = go.Figure()
clients_bar_chart.add_trace(go.Bar(
    x=clients_per_year['Year'],
    y=clients_per_year['Partner_code'],
    text=clients_per_year['Partner_code'].round(2),
    textposition='outside',
    marker_color=colors
))
clients_bar_chart.update_layout(
    title='NC per Year',
    xaxis_title='2022    2023',
    yaxis_title='',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    yaxis=dict(showgrid=False, showticklabels=False),
    xaxis=dict(showgrid=False, showticklabels=False)
)

# Define the colors for the years in repeat rate chart
repeat_colors = ['#FA8072' if year % 2 == 0 else '#800000' for year in repeat_clients_per_year['Year']]

# Create the repeat purchase rate bar chart
repeat_rate_bar_chart = go.Figure()
repeat_rate_bar_chart.add_trace(go.Bar(
    x=repeat_clients_per_year['Year'],
    y=repeat_clients_per_year['Repeat_rate'],
    text=repeat_clients_per_year['Repeat_rate'].round(2),
    textposition='outside',
    marker_color=repeat_colors
))
repeat_rate_bar_chart.update_layout(
    title='RPR per Year',
    xaxis_title='2022    2023',
    yaxis_title='',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    yaxis=dict(showgrid=False, showticklabels=False),
    xaxis=dict(showgrid=False, showticklabels=False)
)

# Create columns with specified relative widths
col_left_1, col_center_1, col_right_1 = st.columns([1, 3, 1])

# Display the map in the center column
with col_center_1:
    st_folium(sales_map)

# Display the number of clients per year bar chart in the left column
with col_left_1:
    st.plotly_chart(clients_bar_chart, use_container_width=True)

# Display the repeat purchase rate bar chart in the right column
with col_right_1:
    st.plotly_chart(repeat_rate_bar_chart, use_container_width=True)



######## Top 10 Products

import matplotlib.pyplot as plt

# Calculate the average discount and total sales for each product
product_analysis = filtered_data.groupby(['Product']).agg({
    'Discount_%': 'mean',
    'Sales': 'sum'
}).reset_index()

# Function to plot the top 10 products with the highest sales and their average discount
def plot_top_sales_products(product_analysis):
    top_sales = product_analysis.nlargest(10, 'Sales')
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), sharey=True)
    
    # Plot the top ten products with the highest sales
    axes[1].barh(top_sales['Product'], top_sales['Sales'], color='#FA8072', edgecolor='black')
    axes[1].set_xlabel('Total Sales')
    axes[1].set_title('Top 10 Products with Highest Sales')
    
    # Plot the average discount of the top ten products with the highest sales
    axes[0].barh(top_sales['Product'], top_sales['Discount_%'], color='black', edgecolor='#FA8072')
    axes[0].set_xlabel('Average Discount (%)')
    axes[0].set_title('Average Discount of Top 10 Products with Highest Sales')
    
    # Remove background and borders
    for ax in axes:
        ax.set_facecolor('#ffdab9')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    
    plt.tight_layout()
    fig.patch.set_facecolor('#ffdab9')  
    st.pyplot(fig)



# Create columns with structure 1, 6, 1
col1, col2, col3 = st.columns([1, 6, 1])


# Display the plot in the middle column
with col2:
    plot_top_sales_products(product_analysis)


########################### Secont Part #########################################
# Ensure Order_date is a datetime column
st.title("Add-Ons")


data_second['Order_date'] = pd.to_datetime(data_second['Order_date'])

# Extract month and year from Order_date
data_second['YearMonth'] = data_second['Order_date'].dt.to_period('M')
data_second['Month'] = data_second['Order_date'].dt.strftime('%b')
data_second['Year'] = data_second['Order_date'].dt.year

# Sort by Month for correct order in the plots
data_second['Month'] = pd.Categorical(data_second['Month'], categories=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], ordered=True)


# Apply the filters to data_second
data_second_filtered = data_second[
    (data_second['Order_date'] >= pd.Timestamp(start_period.start_time)) &
    (data_second['Order_date'] <= pd.Timestamp(end_period.end_time)) &
    (data_second['Customer_country'].isin([selected_country]) if selected_country != "All" else True) &
    (data_second['City'].isin(selected_cities)) &
    (data_second['Distribution_type'].isin(selected_distribution_types))
]


# Pie Chart for Sum of Sales per Product with filtered data
sales_per_product = data_second_filtered.groupby('Product')['Sales'].sum().reset_index()

fig_pie = px.pie(sales_per_product, values='Sales', names='Product', title='Sum of Sales per Product',
                 template='plotly_dark')
fig_pie.update_traces(textinfo='none', hoverinfo='label+percent+value')
fig_pie.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)

color=['black', 'red']
# Line Chart for Count of Orders per Month with filtered data
orders_per_month = data_second_filtered.groupby(['Month', 'Year'], observed=False)['Document_number'].nunique().reset_index()
fig_orders = px.line(orders_per_month, x='Month', y='Document_number', color='Year', title='Count of Orders per Month',
                     template='plotly_dark', color_discrete_sequence=color)
fig_orders.update_layout(
    yaxis_title=None,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)

# Line Chart for Count of Clients per Month with filtered data
clients_per_month = data_second_filtered.groupby(['Month', 'Year'], observed=False)['Partner_code'].nunique().reset_index()
fig_clients = px.line(clients_per_month, x='Month', y='Partner_code', color='Year', title='Count of Clients per Month',
                      template='plotly_dark', color_discrete_sequence=color)
fig_clients.update_layout(
    yaxis_title=None,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)

# Histogram for Count of Products per Month with filtered data
products_per_month = data_second_filtered.groupby(['Month', 'Year'], observed=False)['Product'].nunique().reset_index()
fig_products = px.histogram(products_per_month, x='Month', y='Product', color='Year', title='Count of Products per Month',
                            barmode='overlay', template='plotly_dark', color_discrete_sequence=color)
fig_products.update_traces(opacity=0.6)
fig_products.update_layout(
    yaxis_title=None,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)

# Place pie chart and first line plot in one row
col1_second, col2_second = st.columns(2)
with col1_second:
    st.plotly_chart(fig_pie, use_container_width=True)
with col2_second:
    st.plotly_chart(fig_orders, use_container_width=True)

# Place the other two line plots below them
col3_second, col4_second = st.columns(2)
with col3_second:
    st.plotly_chart(fig_clients, use_container_width=True)
with col4_second:
    st.plotly_chart(fig_products, use_container_width=True)









