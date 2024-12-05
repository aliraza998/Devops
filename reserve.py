import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64
import streamlit.components.v1 as components

# Load models
# Replace with your trained model's path
roi_model = joblib.load("roi_model.pkl")
# Replace with your trained model's path
budget_model = joblib.load("budget_model.pkl")

# Load fitted encoders
le_event = joblib.load("event_encoder.pkl")
le_type = joblib.load("type_encoder.pkl")
le_city = joblib.load("city_encoder.pkl")

# Event_Name and Promotional_Type options
event_names = [
    "Ramadan", "Eid Ul Fitr", "Wedding Season", "National Health Day",
    "Eid Ul Adha", "Kashmir Day", "Muharram", "Labour Day",
    "National Food Day", "Independence Day"
]

promotional_types = [
    "Social Media Campaigns", "Discount offers", "Celebrity Endorsements", "Charity Promotions",
    "TV and Radio Campaigns", "Retail Partnerships", "Sponsorship of Local Events", "Educational Campaigns",
    "In-Store Displays", "Wedding Packages", "Recipe Contests", "Promotions For Kashmir Relief",
    "Collaborations with NGOs", "Special Packaging", "Branding with Religious themes",
    "Collaborations with National Events", "Advertisements", "Partnership with Nutritionists"
]

cities = [
    "Lahore", "Islamabad", "Faisalabad", "Peshawar", "Sukkur", "Multan", "Hyderabad",
    "Larkana", "Nawabshah", "Bahawalpur", "Rahim Yar Khan", "Karachi"
]

# Function to encode the local image


def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode()
    return base64_str


# Paths for images
# Replace with the correct relative path
background_image_path = "images/background.jpg"
sidebar_logo_path = "images/side.png"

# Encode images
base64_image = get_base64_image(background_image_path)
sidebar_logo_image = get_base64_image(sidebar_logo_path)

# Add CSS for background image
page_bg_img = f"""
<style>
/* App-wide background styling */
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{base64_image}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

custom_css = f"""
<style>
/* Load Google Sans font */
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');

/* Sidebar styling */
[data-testid="stSidebar"] {{
    background-color: #8A9A5B; /* Light green background */
}}

/* Sidebar logo positioning */
.side_logo {{
    text-align: center;
    margin-top: -70px;  /* Move the logo higher */
    margin-bottom: 5px;
}}
.side_logo img {{
    width: 200px; /* Adjust the logo size */
    margin: 0 auto;
}}

/* Apply Google Sans font to the entire app */
html, body, [data-testid="stAppViewContainer"], h1, h2, h3, h4, p, div, button {{
    font-family: 'Google Sans', sans-serif;
}}

/* Header styling */
h1, h2, h3, h4 {{
    color: #FFFFFF; /* White text for headers */
}}

/* Button styling */
button {{
    background-color: #013220 !important; /* Green background for buttons */
    color: #ffffff !important; /* White text on buttons */
    border-radius: 5px; /* Rounded button corners */
    padding: 8px;
    border: none;
}}

/* Adjust button hover effects */
button:hover {{
    background-color: #1f6d34 !important; /* Darker green on hover */
}}

/* Sidebar label color styling */
[data-testid="stSidebar"] label {{
    color: #FFFFFF !important;  /* Make sure the text color is white for all labels */
}}

[data-testid="stSidebar"] .st-radio label {{
    color: #FFFFFF !important;  /* Change color of radio options labels */
}}

[data-testid="stSidebar"] .st-radio div {{
    color: #FFFFFF !important;  /* Ensure radio options (both label and div) are white */
}}

[data-testid="stSidebar"] .st-radio input {{
    accent-color: #FFFFFF !important;  /* Change the radio button itself to white */
}}

[data-testid="stSidebar"] .st-selectbox label, 
[data-testid="stSidebar"] .stNumberInput label {{
    color: #FFFFFF !important;  /* Specifically target these widget labels */
}}

/* Ensuring radio options inside the sidebar are styled properly */
[data-testid="stSidebar"] .st-radio div[data-baseweb="radio"] {{
    color: #FFFFFF !important;  /* Ensures radio button options in sidebar are white */
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Display logo in the sidebar
st.sidebar.markdown(f"""
    <div class="side_logo">
        <img src="data:image/png;base64,{sidebar_logo_image}" alt="Sidebar Logo">
    </div>
""", unsafe_allow_html=True)

# Sidebar for input features
st.sidebar.header("Filters")
event_name = st.sidebar.selectbox("Event Name", event_names)
promotional_type = st.sidebar.selectbox("Promotional Type", promotional_types)
# Replaced text_input with selectbox
city = st.sidebar.selectbox("City", cities)
sales_revenue = st.sidebar.number_input(
    "Sales Revenue", min_value=0.0, step=100.0)
selected_option = st.sidebar.radio(
    "What to Predict?", ("ROI", "Investment"))

if selected_option == "ROI":
    promotional_budget = st.sidebar.number_input(
        "Investment", min_value=0.0, step=100.0)
elif selected_option == "Investment":
    roi = st.sidebar.number_input("ROI", min_value=0.0, step=0.1)

# Load dataset for historical data visualization
df = pd.read_csv("dataset.csv")  # Updated to your dataset name

# Initialize prediction variable
prediction = None
x_label = "Investment"  # Default label for x-axis
y_label = "ROI"  # Default label for y-axis


st.markdown("<br>" * 2, unsafe_allow_html=True)
# Filter historical data based on current selections
filtered_data = df[
    (df["Event_Name"] == event_name) &
    (df["Promotional_Type"] == promotional_type) &
    (df["City"] == city)
]

# Convert 'Date' to datetime
filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])

# Resample to aggregate data by week or month
filtered_data.set_index('Date', inplace=True)
resampled_data = filtered_data.resample('W').mean(
    numeric_only=True).reset_index()  # Weekly aggregation

# Interpolation to smooth data
resampled_data['ROI'] = resampled_data['ROI'].interpolate(method='linear')
resampled_data['Promotional_Budget'] = resampled_data['Promotional_Budget'].interpolate(
    method='linear')

# Plotting
x_values = resampled_data['Date']
if selected_option == "ROI":
    y_values = resampled_data['ROI']
    label_y = 'ROI'
elif selected_option == "Investment":
    y_values = resampled_data['Promotional_Budget']
    label_y = 'Investment'

# Create plot with custom base color (light gray background for the graph)
fig, ax = plt.subplots(figsize=(20, 8), facecolor='#BCB88A')  # Light gray background color for the figure

# Set background color for the axes (plot area)
ax.set_facecolor('#BCB88A')  # Slightly darker gray background for the plot area

# Plot historical trendline with blue color
ax.plot(x_values, y_values, label=f'{label_y} Trendline', color='#013220', linewidth=2)

# Plot predicted value as a horizontal line with blue color (for both ROI and Investment predictions)
if prediction:
    ax.axhline(y=prediction, color='blue', linestyle='--', linewidth=2, label=f'Predicted {label_y}')

    # Annotate the predicted value for better understanding
    ax.annotate(
        f"Predicted {label_y}: {prediction:.2f}",
        xy=(x_values.iloc[-1], prediction),  # Position annotation at the end of the x-axis
        xytext=(x_values.iloc[-1], prediction + 0.05),  # Adjust annotation slightly above the line
        fontsize=10,
        color='blue',
        arrowprops=dict(facecolor='blue', arrowstyle='->')
    )

# Formatting
ax.set_xlabel("Date", fontsize=20, color='#013220')  # Set x-axis label color to green
ax.set_ylabel(label_y, fontsize=20, color='#013220')  # Set y-axis label color to green
ax.set_title(f"{label_y} Trend Over Time for {event_name} in {city} ({promotional_type})", fontsize=20, color='#013220')  # Title in green

# Set grid and adjust its properties
ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

# Set legend background and text color
ax.legend(facecolor='#FFFFFF', edgecolor='white', fontsize=15, loc='upper right')

# Adjust x-axis formatting for better readability
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Major ticks at every 6 months
ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Minor ticks for monthly intervals
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format as "Month Year"
plt.xticks(rotation=45, fontsize=15)  # Increase font size for x-axis labels
plt.yticks(rotation=0, fontsize=15)
# Display plot
st.pyplot(fig)




# Prediction
if st.sidebar.button("Predict"):
    try:
        # Encode inputs using LabelEncoders
        event_name_encoded = le_event.transform([event_name])[0]
        promotional_type_encoded = le_type.transform([promotional_type])[0]
        city_encoded = le_city.transform([city])[0]

        if selected_option == "ROI":
            # Prepare input data for ROI prediction
            input_data = pd.DataFrame({
                "Event_Name": [event_name_encoded],
                "Promotional_Budget": [promotional_budget],
                "Promotional_Type": [promotional_type_encoded],
                "City": [city_encoded],
                "Sales_Revenue": [sales_revenue]
            })

            # Predict ROI
            prediction = roi_model.predict(input_data)[0]
            st.markdown(
                f"<h3 style='color:#8A9A5B;margin-top:10px; text-align: center;'>Predicted ROI: {prediction:.2f}</h3>", unsafe_allow_html=True)
            x_label = "Investment"
            y_label = "ROI"

        elif selected_option == "Investment":
            # Prepare input data for Investment prediction
            input_data = pd.DataFrame({
                "Event_Name": [event_name_encoded],
                "ROI": [roi],
                "Promotional_Type": [promotional_type_encoded],
                "City": [city_encoded],
                "Sales_Revenue": [sales_revenue]
            })

            # Predict Investment
            prediction = budget_model.predict(input_data)[0]
            st.markdown(
                f"<h3 style='color:#8A9A5B;margin-top:10px; text-align: center;'>Predicted Investment: {prediction:.2f}</h3>", unsafe_allow_html=True)

            # Check prediction range
            if prediction < 0:
                st.warning(
                    f"Warning: The predicted Investment seems out of expected range: {prediction}.")

            x_label = "ROI"
            y_label = "Investment"

    except Exception as e:
        st.error(f"Error: {str(e)}")


# Title for your Streamlit app
st.title("Power BI Dashboard Embed")

# Embed Power BI report using iframe
power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=f85c3ce5-e5b6-4275-bfea-05fc5c98ff68&appId=2b99fecb-2377-4f75-8496-792db7e2b66a&autoAuth=true&ctid=8d513303-4b89-4180-b233-bebb388ad37f"  # Replace with your Power BI embed link
components.iframe(power_bi_url, width=1000, height=600)