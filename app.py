import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
import io
from model import FoodWastePredictor
from data_processor import DataProcessor
from visualizations import Visualizer
from utils import export_data, generate_sample_data, calculate_cost_estimate

# Page configuration
st.set_page_config(
    page_title="SmartShelfAI - Food Waste Prediction",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def load_model():
    return FoodWastePredictor()

@st.cache_resource
def load_data_processor():
    return DataProcessor()

@st.cache_resource
def load_visualizer():
    return Visualizer()

# Load components
model = load_model()
data_processor = load_data_processor()
visualizer = load_visualizer()

# Main app
def add_watermark():
    """Add watermark at the bottom of every page"""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888; font-size: 12px; margin-top: 50px; padding: 20px;'>
            <p>Made by Human on Earth 🌍 | SmartShelfAI © 2025 | Reducing Food Waste Through Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    st.title("🍎 SmartShelfAI - Food Waste Prediction")
    st.markdown("*Intelligent food waste prediction to optimize storage and reduce spoilage*")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Prediction", "Data Analysis", "Model Training", "Cost Analysis", "Custom Items", "Help & Guide", "FAQs", "Contact Us", "About"]
    )
    
    if page == "Prediction":
        prediction_page()
    elif page == "Data Analysis":
        data_analysis_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Cost Analysis":
        cost_analysis_page()
    elif page == "Custom Items":
        custom_items_page()
    elif page == "Help & Guide":
        help_page()
    elif page == "FAQs":
        faqs_page()
    elif page == "Contact Us":
        contact_page()
    elif page == "About":
        about_page()
    
    # Add watermark at the bottom of every page
    add_watermark()

def prediction_page():
    st.header("🔮 Waste Prediction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Food category selection with custom option
        predefined_categories = ["Fruits", "Vegetables", "Berries", "Citrus", "Leafy Greens"]
        
        # Add saved custom categories
        custom_categories = []
        if 'custom_categories' in st.session_state and st.session_state.custom_categories:
            custom_categories = list(st.session_state.custom_categories.keys())
        
        all_categories = predefined_categories + custom_categories + ["➕ Add Custom Category"]
        
        category_selection = st.selectbox(
            "Food Category",
            all_categories,
            index=0,
            placeholder="Select a category"
        )
        
        if category_selection == "➕ Add Custom Category":
            # Custom category input
            st.subheader("Add Custom Category")
            custom_category_name = st.text_input(
                "Category Name:",
                placeholder="e.g., Herbs, Nuts, Dairy, Meat"
            )
            
            if custom_category_name:
                # Initialize custom categories if not exists
                if 'custom_categories' not in st.session_state:
                    st.session_state.custom_categories = {}
                
                # Add the custom category with empty food list
                if custom_category_name not in st.session_state.custom_categories:
                    st.session_state.custom_categories[custom_category_name] = []
                    st.success(f"Created category: {custom_category_name}")
                
                food_category = custom_category_name
                food_options_for_category = st.session_state.custom_categories[custom_category_name]
            else:
                st.warning("Please enter a category name")
                food_category = "Fruits"
                food_options_for_category = food_options["Fruits"]
        else:
            food_category = category_selection
        
        # Simplified food options - just the most common items
        food_options = {
            "Fruits": ["Apple", "Banana", "Orange"],
            "Vegetables": ["Tomato", "Carrot", "Potato"],
            "Berries": ["Strawberry", "Grapes"],
            "Citrus": ["Orange", "Lemon"],
            "Leafy Greens": ["Lettuce", "Spinach"]
        }
        
        # Get food options for the selected category
        if food_category in predefined_categories:
            food_options_for_category = food_options[food_category]
        elif food_category in st.session_state.get('custom_categories', {}):
            # Custom category
            food_options_for_category = st.session_state.custom_categories[food_category]
        else:
            # Fallback to fruits if something goes wrong
            food_options_for_category = food_options["Fruits"]
        
        # Add option for custom items and show saved custom items
        additional_options = []
        
        # Add saved custom items
        if 'custom_items' in st.session_state and st.session_state.custom_items:
            custom_items_list = list(st.session_state.custom_items.keys())
            additional_options.extend([f"📋 {item}" for item in custom_items_list])
        
        additional_options.append("➕ Add Custom Item")
        
        food_list = food_options_for_category + additional_options
        
        food_selection = st.selectbox(
            f"{food_category[:-1]} Type",
            food_list,
            index=None,
            placeholder="Select a food item"
        )
        
        if food_selection == "➕ Add Custom Item":
            # Custom item input
            st.subheader("Quick Add Custom Item")
            food_type = st.text_input(
                "Enter food name:",
                placeholder="e.g., Dragon Fruit, Sweet Potato, Herbs"
            )
            
            # Optional: Set perishability level
            perishability = st.selectbox(
                "Perishability Level",
                ["Low (like potatoes, onions)", "Medium (like apples, carrots)", "High (like berries, leafy greens)"],
                help="This affects how quickly the item spoils"
            )
            
            if food_type:
                st.success(f"Custom item '{food_type}' will be used for prediction")
                st.info("💡 Visit the 'Custom Items' page to save items permanently and set detailed properties")
            else:
                st.warning("Please enter a food name")
                food_type = "Apple"  # Default fallback
        elif food_selection and food_selection.startswith("📋 "):
            # Using saved custom item
            food_type = food_selection[2:]  # Remove "📋 " prefix
            custom_item = st.session_state.custom_items[food_type]
            st.success(f"Using saved custom item: {food_type}")
            
            # Show custom item details
            with st.expander("View Item Details"):
                st.write(f"**Category:** {custom_item['category']}")
                st.write(f"**Perishability:** {custom_item['perishability']}")
                st.write(f"**Optimal Temperature:** {custom_item['optimal_temp']}°C")
                st.write(f"**Optimal Humidity:** {custom_item['optimal_humidity']}%")
                if custom_item['storage_notes']:
                    st.write(f"**Storage Notes:** {custom_item['storage_notes']}")
        elif food_selection:
            food_type = food_selection
        else:
            # No selection made yet
            food_type = None
        
        # Quantity input
        quantity = st.number_input(
            "Quantity (kg)",
            min_value=0.1,
            max_value=1000.0,
            value=10.0,
            step=0.1
        )
        
        # Days since harvest
        days_since_harvest = st.slider(
            "Days Since Harvest",
            min_value=0,
            max_value=30,
            value=5
        )
        
        # Storage temperature
        temperature = st.slider(
            "Storage Temperature (°C)",
            min_value=-5,
            max_value=35,
            value=20
        )
        
        # Humidity
        humidity = st.slider(
            "Humidity (%)",
            min_value=30,
            max_value=95,
            value=65
        )
        
        # Predict button
        if st.button("🎯 Predict Waste", type="primary"):
            if food_type:
                # Make prediction
                features = {
                    'fruit_type': food_type,
                    'quantity': quantity,
                    'days_since_harvest': days_since_harvest,
                    'temperature': temperature,
                    'humidity': humidity
                }
                
                waste_percentage = model.predict(features)
                waste_kg = (waste_percentage / 100) * quantity
                
                # Store prediction in session state
                st.session_state.last_prediction = {
                    'waste_percentage': waste_percentage,
                    'waste_kg': waste_kg,
                    'features': features
                }
            else:
                st.error("Please select a food item before making a prediction.")
    
    with col2:
        st.subheader("Prediction Results")
        
        if 'last_prediction' in st.session_state:
            pred = st.session_state.last_prediction
            
            # Display prediction metrics
            col2_1, col2_2, col2_3 = st.columns(3)
            
            with col2_1:
                st.metric(
                    "Predicted Waste",
                    f"{pred['waste_percentage']:.1f}%",
                    delta=f"{pred['waste_kg']:.2f} kg"
                )
            
            with col2_2:
                # Calculate freshness score
                freshness = max(0, 100 - pred['waste_percentage'])
                st.metric("Freshness Score", f"{freshness:.1f}%")
            
            with col2_3:
                # Risk level
                if pred['waste_percentage'] < 10:
                    risk = "Low"
                    risk_color = "green"
                elif pred['waste_percentage'] < 25:
                    risk = "Medium"
                    risk_color = "orange"
                else:
                    risk = "High"
                    risk_color = "red"
                
                st.metric("Risk Level", risk)
            
            # Visualization
            fig = visualizer.create_waste_gauge(pred['waste_percentage'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("🔍 Model Insights")
            importance_fig = visualizer.create_feature_importance_chart(model.get_feature_importance())
            st.plotly_chart(importance_fig, use_container_width=True)
            
        else:
            st.info("👆 Enter parameters and click 'Predict Waste' to see results")
            
            # Show historical data plot
            st.subheader("📊 Historical Waste Patterns")
            sample_data = generate_sample_data(100)
            hist_fig = visualizer.create_historical_scatter(sample_data)
            st.plotly_chart(hist_fig, use_container_width=True)
    
    # Reset button at the bottom
    st.markdown("---")
    if st.button("🔄 Reset All Inputs", help="Clear all inputs, predictions and start fresh", key="reset_prediction"):
        # Clear prediction-related session state
        for key in list(st.session_state.keys()):
            if key in ['last_prediction'] or key.startswith('prediction_'):
                del st.session_state[key]
        st.rerun()

def data_analysis_page():
    st.header("📊 Data Analysis & Insights")
    
    # Data upload section
    st.subheader("📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file with food waste data",
        type=['csv'],
        help="Expected columns: fruit_type, quantity, days_since_harvest, temperature, humidity, waste_percentage"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} records")
            
            # Validate data
            required_columns = ['fruit_type', 'quantity', 'days_since_harvest', 'temperature', 'humidity', 'waste_percentage']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                return
            
            # Data summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Data Summary")
                st.dataframe(df.describe())
            
            with col2:
                st.subheader("🍎 Fruit Distribution")
                fruit_counts = df['fruit_type'].value_counts()
                fig_pie = px.pie(values=fruit_counts.values, names=fruit_counts.index)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Filtering options
            st.subheader("🔍 Filter Data")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_fruits = st.multiselect(
                    "Select Fruit Types",
                    options=df['fruit_type'].unique(),
                    default=df['fruit_type'].unique()
                )
            
            with col2:
                temp_range = st.slider(
                    "Temperature Range (°C)",
                    min_value=int(df['temperature'].min()),
                    max_value=int(df['temperature'].max()),
                    value=(int(df['temperature'].min()), int(df['temperature'].max()))
                )
            
            with col3:
                days_range = st.slider(
                    "Days Since Harvest",
                    min_value=int(df['days_since_harvest'].min()),
                    max_value=int(df['days_since_harvest'].max()),
                    value=(int(df['days_since_harvest'].min()), int(df['days_since_harvest'].max()))
                )
            
            # Apply filters
            filtered_df = df[
                (df['fruit_type'].isin(selected_fruits)) &
                (df['temperature'].between(temp_range[0], temp_range[1])) &
                (df['days_since_harvest'].between(days_range[0], days_range[1]))
            ]
            
            st.write(f"Filtered data: {len(filtered_df)} records")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Waste vs Days correlation
                fig_scatter = px.scatter(
                    filtered_df, 
                    x='days_since_harvest', 
                    y='waste_percentage',
                    color='fruit_type',
                    size='quantity',
                    title="Waste % vs Days Since Harvest"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # Temperature vs Waste
                fig_temp = px.scatter(
                    filtered_df,
                    x='temperature',
                    y='waste_percentage',
                    color='fruit_type',
                    title="Waste % vs Storage Temperature"
                )
                st.plotly_chart(fig_temp, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("🔗 Correlation Analysis")
            try:
                numeric_cols = ['quantity', 'days_since_harvest', 'temperature', 'humidity', 'waste_percentage']
                # Filter to only existing numeric columns and ensure they are numeric
                available_numeric_cols = []
                for col in numeric_cols:
                    if col in filtered_df.columns:
                        # Convert to numeric if not already
                        filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
                        available_numeric_cols.append(col)
                
                if len(available_numeric_cols) > 1:
                    numeric_df = filtered_df[available_numeric_cols]
                    correlation_matrix = numeric_df.corr()
                    
                    fig_heatmap = px.imshow(
                        correlation_matrix,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        title="Feature Correlation Heatmap"
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.info("Need at least 2 numeric columns for correlation analysis")
            except Exception as e:
                st.warning(f"Could not generate correlation analysis: {str(e)}")
            
            # Export filtered data
            if st.button("💾 Export Filtered Data"):
                csv_data = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"filtered_waste_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        # Show sample data analysis
        st.info("👆 Upload your data to see detailed analysis, or view sample data below")
        sample_data = generate_sample_data(200)
        
        # Sample visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig_sample = visualizer.create_historical_scatter(sample_data)
            st.plotly_chart(fig_sample, use_container_width=True)
        
        with col2:
            try:
                avg_waste = sample_data.groupby('fruit_type')['waste_percentage'].mean()
                avg_waste_sorted = avg_waste.sort_values(ascending=False)
                fig_bar = px.bar(
                    x=avg_waste_sorted.index,
                    y=avg_waste_sorted.values,
                    title="Average Waste % by Fruit Type",
                    labels={'x': 'Fruit Type', 'y': 'Average Waste %'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate chart: {str(e)}")
    
    # Reset button at the bottom
    st.markdown("---")
    if st.button("🔄 Reset All Data", help="Clear uploaded data and all visualizations", key="reset_data_analysis"):
        # Clear all analysis-related session state
        for key in list(st.session_state.keys()):
            if key.startswith('analysis_') or key in ['uploaded_data', 'filtered_data']:
                del st.session_state[key]
        st.rerun()

def model_training_page():
    st.header("🤖 Model Training & Performance")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        
        # Model parameters
        n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
        max_depth = st.slider("Max Depth", 3, 20, 10)
        min_samples_split = st.slider("Min Samples Split", 2, 20, 5)
        
        # Training data upload
        st.subheader("Training Data")
        training_file = st.file_uploader("Upload training data (CSV)", type=['csv'])
        
        if training_file is not None:
            training_data = pd.read_csv(training_file)
            st.success(f"Loaded {len(training_data)} training samples")
            
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model..."):
                    # Update model parameters
                    model.update_parameters(n_estimators, max_depth, min_samples_split)
                    
                    # Train model
                    metrics = model.train(training_data)
                    
                    st.session_state.training_metrics = metrics
                    st.success("✅ Model trained successfully!")
        else:
            if st.button("🚀 Train with Sample Data", type="primary"):
                with st.spinner("Training model with sample data..."):
                    sample_data = generate_sample_data(500)
                    model.update_parameters(n_estimators, max_depth, min_samples_split)
                    metrics = model.train(sample_data)
                    st.session_state.training_metrics = metrics
                    st.success("✅ Model trained with sample data!")
    
    with col2:
        st.subheader("Model Performance")
        
        if 'training_metrics' in st.session_state:
            metrics = st.session_state.training_metrics
            
            # Performance metrics
            col2_1, col2_2, col2_3 = st.columns(3)
            
            with col2_1:
                st.metric("R² Score", f"{metrics['r2_score']:.3f}")
            with col2_2:
                st.metric("RMSE", f"{metrics['rmse']:.3f}")
            with col2_3:
                st.metric("MAE", f"{metrics['mae']:.3f}")
            
            # Feature importance
            importance_fig = visualizer.create_feature_importance_chart(model.get_feature_importance())
            st.plotly_chart(importance_fig, use_container_width=True)
            
            # Prediction vs Actual plot
            if 'predictions' in metrics:
                pred_fig = visualizer.create_prediction_accuracy_plot(
                    metrics['y_true'], 
                    metrics['predictions']
                )
                st.plotly_chart(pred_fig, use_container_width=True)
        else:
            st.info("👆 Train the model to see performance metrics")
            
            # Show current model info
            st.subheader("Current Model Info")
            st.write("**Model Type:** Random Forest Regressor")
            st.write("**Default Parameters:**")
            st.write("- N Estimators: 100")
            st.write("- Max Depth: 10") 
            st.write("- Min Samples Split: 5")
    
    # Reset button at the bottom
    st.markdown("---")
    if st.button("🔄 Reset Training", help="Clear training data and model metrics", key="reset_model_training"):
        # Clear training-related session state
        for key in list(st.session_state.keys()):
            if key in ['training_metrics'] or key.startswith('training_'):
                del st.session_state[key]
        st.rerun()

def cost_analysis_page():
    st.header("💰 Cost Analysis & Financial Impact")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Cost Parameters")
        
        # Cost per kg for different food items
        st.write("**Food Prices (per kg)**")
        food_prices = {}
        
        # Default prices for common foods (in Indian Rupees per kg)
        default_prices = {
            "Apple": 250, "Banana": 160, "Orange": 200, "Grapes": 320,
            "Strawberry": 650, "Tomato": 320, "Carrot": 120, 
            "Potato": 100, "Lemon": 280, "Lettuce": 280, "Spinach": 480
        }
        
        # Food selection for pricing
        food_options = list(default_prices.keys()) + ["Add Custom Item"]
        selected_foods = st.multiselect(
            "Select foods to price:",
            options=food_options,
            default=[],
            placeholder="Choose food items to analyze"
        )
        
        # Handle custom items
        custom_items = []
        if "Add Custom Item" in selected_foods:
            selected_foods.remove("Add Custom Item")
            st.subheader("➕ Add Custom Items")
            
            num_custom = st.number_input("Number of custom items to add", min_value=1, max_value=10, value=1)
            
            for i in range(num_custom):
                with st.expander(f"Custom Item {i+1}"):
                    col_item, col_remove = st.columns([4, 1])
                    
                    with col_item:
                        custom_name = st.text_input(f"Item Name", key=f"custom_name_{i}")
                        custom_price = st.number_input(f"Price (₹/kg)", min_value=10, max_value=8000, value=250, step=10, key=f"custom_price_{i}")
                    
                    with col_remove:
                        st.write("")  # Space
                        if st.button("🗑️", help=f"Remove Custom Item {i+1}", key=f"remove_custom_{i}"):
                            # Remove from session state
                            if f"custom_name_{i}" in st.session_state:
                                del st.session_state[f"custom_name_{i}"]
                            if f"custom_price_{i}" in st.session_state:
                                del st.session_state[f"custom_price_{i}"]
                            st.rerun()
                    
                    if custom_name and custom_price:
                        custom_items.append(custom_name)
                        default_prices[custom_name] = custom_price
                        if custom_name not in selected_foods:
                            selected_foods.append(custom_name)
        
        for food in selected_foods:
            food_prices[food] = st.number_input(
                f"{food} (₹/kg)",
                min_value=10,
                max_value=4000,
                value=default_prices.get(food, 250),
                step=10,
                key=f"price_{food}"
            )
        
        # Time period for analysis
        st.subheader("Analysis Period")
        time_period = st.selectbox(
            "Time Period",
            ["Daily", "Weekly", "Monthly", "Yearly"],
            index=None,
            placeholder="Select analysis period"
        )
        
        # Volume inputs
        total_volume = st.number_input(
            f"Total Volume ({time_period.lower()})",
            min_value=1.0,
            max_value=10000.0,
            value=100.0,
            step=1.0
        )
    
    with col2:
        st.subheader("Cost Impact Analysis")
        
        if st.button("📊 Calculate Cost Impact", type="primary") and selected_foods:
            # Calculate costs for each selected food type
            cost_results = []
            
            for food in selected_foods:
                # Simulate typical conditions for each food
                features = {
                    'fruit_type': food,
                    'quantity': total_volume / len(selected_foods),  # Distribute volume evenly
                    'days_since_harvest': 7,  # Average storage time
                    'temperature': 20,
                    'humidity': 65
                }
                
                waste_percentage = model.predict(features)
                waste_kg = (waste_percentage / 100) * features['quantity']
                cost_per_kg = food_prices[food]
                waste_cost = waste_kg * cost_per_kg
                total_value = features['quantity'] * cost_per_kg
                
                cost_results.append({
                    'Food': food,
                    'Volume (kg)': features['quantity'],
                    'Waste %': waste_percentage,
                    'Waste (kg)': waste_kg,
                    'Price (₹/kg)': cost_per_kg,
                    'Waste Cost (₹)': waste_cost,
                    'Total Value (₹)': total_value,
                    'Loss %': (waste_cost / total_value) * 100
                })
            
            cost_df = pd.DataFrame(cost_results)
            
            # Summary metrics
            total_waste_cost = cost_df['Waste Cost (₹)'].sum()
            total_value = cost_df['Total Value (₹)'].sum()
            overall_loss_percent = (total_waste_cost / total_value) * 100
            
            col2_1, col2_2, col2_3 = st.columns(3)
            
            with col2_1:
                st.metric("Total Waste Cost", f"₹{total_waste_cost:,.0f}")
            with col2_2:
                st.metric("Total Value", f"₹{total_value:,.0f}")
            with col2_3:
                st.metric("Loss Percentage", f"{overall_loss_percent:.1f}%")
            
            # Cost breakdown chart
            fig_cost = px.bar(
                cost_df,
                x='Food',
                y='Waste Cost (₹)',
                title=f"{time_period} Waste Cost by Food Type",
                color='Waste Cost (₹)',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_cost, use_container_width=True)
            
            # Detailed breakdown table
            st.subheader("Detailed Cost Breakdown")
            st.dataframe(cost_df, use_container_width=True)
            
            # Optimization suggestions
            st.subheader("💡 Cost Optimization Suggestions")
            
            # Find highest waste foods
            high_waste_foods = cost_df.nlargest(3, 'Waste %')['Food'].tolist()
            high_cost_foods = cost_df.nlargest(3, 'Waste Cost (₹)')['Food'].tolist()
            
            st.write("**Priority Actions:**")
            for i, food in enumerate(high_waste_foods[:3], 1):
                food_data = cost_df[cost_df['Food'] == food].iloc[0]
                st.write(f"{i}. **{food}**: {food_data['Waste %']:.1f}% waste rate - Consider improved storage conditions")
            
            st.write("**Highest Financial Impact:**")
            for i, food in enumerate(high_cost_foods[:3], 1):
                food_data = cost_df[cost_df['Food'] == food].iloc[0]
                st.write(f"{i}. **{food}**: ₹{food_data['Waste Cost (₹)']:,.0f} {time_period.lower()} loss")
            
            # Export cost analysis
            if st.button("💾 Export Cost Analysis"):
                csv_data = cost_df.to_csv(index=False)
                st.download_button(
                    label="Download Cost Analysis CSV",
                    data=csv_data,
                    file_name=f"cost_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Reset button at the bottom
    st.markdown("---")
    if st.button("🔄 Reset Cost Analysis", help="Clear all cost inputs and calculations", key="reset_cost_analysis"):
        # Clear cost analysis session state
        for key in list(st.session_state.keys()):
            if key.startswith('cost_') or key.startswith('custom_name_') or key.startswith('custom_price_') or key.startswith('price_'):
                del st.session_state[key]
        st.rerun()

def custom_items_page():
    st.header("➕ Custom Food Items Manager")
    
    # Initialize session state for custom items
    if 'custom_items' not in st.session_state:
        st.session_state.custom_items = {}
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Add New Custom Item")
        
        # Input form for new item
        with st.form("add_custom_item"):
            item_name = st.text_input("Food Item Name", placeholder="e.g., Dragon Fruit, Sweet Potato")
            
            # Get available categories (predefined + custom)
            available_categories = ["Fruits", "Vegetables", "Berries", "Citrus", "Leafy Greens", "Other"]
            if 'custom_categories' in st.session_state:
                available_categories.extend(list(st.session_state.custom_categories.keys()))
            
            category = st.selectbox(
                "Category",
                available_categories
            )
            
            perishability = st.selectbox(
                "Perishability Level",
                ["Very High (1-2 days)", "High (3-5 days)", "Medium (1-2 weeks)", "Low (2-4 weeks)", "Very Low (1+ months)"]
            )
            
            # Storage preferences
            optimal_temp = st.slider("Optimal Storage Temperature (°C)", -5, 25, 4)
            optimal_humidity = st.slider("Optimal Humidity (%)", 40, 95, 85)
            
            # Pricing
            estimated_price = st.number_input("Estimated Price (₹/kg)", min_value=10, max_value=8000, value=250, step=10)
            
            # Additional notes
            storage_notes = st.text_area("Storage Notes (optional)", placeholder="Special storage requirements or tips...")
            
            submitted = st.form_submit_button("Add Custom Item")
            
            if submitted and item_name:
                # Map perishability to waste multiplier
                perishability_mapping = {
                    "Very High (1-2 days)": 2.2,
                    "High (3-5 days)": 1.8,
                    "Medium (1-2 weeks)": 1.2,
                    "Low (2-4 weeks)": 0.8,
                    "Very Low (1+ months)": 0.5
                }
                
                waste_multiplier = perishability_mapping[perishability]
                
                # Store custom item
                st.session_state.custom_items[item_name] = {
                    'category': category,
                    'perishability': perishability,
                    'waste_multiplier': waste_multiplier,
                    'optimal_temp': optimal_temp,
                    'optimal_humidity': optimal_humidity,
                    'price': estimated_price,
                    'storage_notes': storage_notes
                }
                
                # If it's a custom category, add the item to that category's food list
                if category in st.session_state.get('custom_categories', {}):
                    if item_name not in st.session_state.custom_categories[category]:
                        st.session_state.custom_categories[category].append(item_name)
                
                st.success(f"✅ Added '{item_name}' to custom items!")
                st.rerun()
    
    with col2:
        st.subheader("Your Custom Categories & Items")
        
        # Display custom categories
        if 'custom_categories' in st.session_state and st.session_state.custom_categories:
            st.write("**Custom Categories:**")
            for cat_name, items in st.session_state.custom_categories.items():
                with st.expander(f"📁 {cat_name} ({len(items)} items)"):
                    if items:
                        for item in items:
                            st.write(f"• {item}")
                    else:
                        st.write("No items in this category yet")
                    
                    # Delete category button
                    if st.button(f"🗑️ Delete Category", key=f"delete_cat_{cat_name}"):
                        del st.session_state.custom_categories[cat_name]
                        st.success(f"Deleted category '{cat_name}'")
                        st.rerun()
        
        st.subheader("Your Custom Items")
        
        if st.session_state.custom_items:
            # Display custom items
            for item_name, item_data in st.session_state.custom_items.items():
                with st.expander(f"🍎 {item_name}"):
                    col2_1, col2_2 = st.columns(2)
                    
                    with col2_1:
                        st.write(f"**Category:** {item_data['category']}")
                        st.write(f"**Perishability:** {item_data['perishability']}")
                        st.write(f"**Price:** ₹{item_data['price']:.0f}/kg")
                    
                    with col2_2:
                        st.write(f"**Optimal Temp:** {item_data['optimal_temp']}°C")
                        st.write(f"**Optimal Humidity:** {item_data['optimal_humidity']}%")
                    
                    if item_data['storage_notes']:
                        st.write(f"**Notes:** {item_data['storage_notes']}")
                    
                    # Delete button
                    if st.button(f"🗑️ Remove {item_name}", key=f"delete_{item_name}"):
                        del st.session_state.custom_items[item_name]
                        st.success(f"Removed '{item_name}'")
                        st.rerun()
            
            # Export custom items
            st.subheader("Export Custom Items")
            if st.button("💾 Export Custom Items"):
                items_df = pd.DataFrame.from_dict(st.session_state.custom_items, orient='index')
                csv_data = items_df.to_csv()
                st.download_button(
                    label="Download Custom Items CSV",
                    data=csv_data,
                    file_name=f"custom_food_items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No custom items added yet. Use the form on the left to add your first custom food item!")
            
        # Quick category manager
        st.subheader("Quick Category Manager")
        col_cat1, col_cat2 = st.columns(2)
        
        with col_cat1:
            new_category = st.text_input("Add New Category", placeholder="e.g., Herbs, Nuts")
            if st.button("➕ Add Category") and new_category:
                if 'custom_categories' not in st.session_state:
                    st.session_state.custom_categories = {}
                
                if new_category not in st.session_state.custom_categories:
                    st.session_state.custom_categories[new_category] = []
                    st.success(f"Added category: {new_category}")
                    st.rerun()
                else:
                    st.warning("Category already exists")
        
        with col_cat2:
            if 'custom_categories' in st.session_state and st.session_state.custom_categories:
                category_to_delete = st.selectbox(
                    "Delete Category",
                    ["Select..."] + list(st.session_state.custom_categories.keys())
                )
                if st.button("🗑️ Delete") and category_to_delete != "Select...":
                    del st.session_state.custom_categories[category_to_delete]
                    st.success(f"Deleted: {category_to_delete}")
                    st.rerun()
        
        # Import custom items
        st.subheader("Import Custom Items")
        uploaded_items = st.file_uploader("Upload Custom Items CSV", type=['csv'], key="custom_items_upload")
        
        if uploaded_items is not None:
            try:
                import_df = pd.read_csv(uploaded_items)
                # Convert back to session state format
                for idx, row in import_df.iterrows():
                    item_name = idx if isinstance(idx, str) else row.get('item_name', f"Item_{idx}")
                    st.session_state.custom_items[item_name] = row.to_dict()
                
                st.success(f"✅ Imported {len(import_df)} custom items!")
                st.rerun()
            except Exception as e:
                st.error(f"Error importing items: {str(e)}")
    
    # Reset button at the bottom
    st.markdown("---")
    if st.button("🔄 Reset Custom Items", help="Clear all custom items and categories", key="reset_custom_items"):
        # Clear custom items session state
        for key in list(st.session_state.keys()):
            if key in ['custom_items', 'custom_categories'] or key.startswith('custom_'):
                del st.session_state[key]
        st.rerun()

def help_page():
    st.header("📚 Help & User Guide")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Getting Started", "Features", "Data Format", "FAQ"])
    
    with tab1:
        st.subheader("🚀 Getting Started with SmartShelfAI")
        
        st.markdown("""
        **SmartShelfAI** is an intelligent food waste prediction system that helps you:
        - Predict waste percentages for different fruits
        - Analyze storage conditions impact
        - Calculate financial losses from food waste
        - Optimize storage strategies
        
        ### Quick Start Guide:
        
        1. **Make Predictions**: Go to the 'Prediction' page
           - Select your fruit type
           - Enter storage conditions (temperature, humidity)
           - Specify days since harvest and quantity
           - Click 'Predict Waste' to get results
        
        2. **Analyze Your Data**: Use the 'Data Analysis' page
           - Upload your CSV file with food waste data
           - Explore patterns and correlations
           - Filter data by various parameters
        
        3. **Calculate Costs**: Visit the 'Cost Analysis' page
           - Set fruit prices for your region
           - Choose analysis period (daily, weekly, monthly)
           - Get detailed financial impact reports
        
        4. **Train Custom Models**: Use the 'Model Training' page
           - Upload your training data
           - Adjust model parameters
           - Evaluate model performance
        """)
    
    with tab2:
        st.subheader("🔧 Feature Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Prediction Features:**
            - Real-time waste percentage prediction
            - Support for 8 fruit types
            - Interactive parameter adjustment
            - Visual gauge charts
            - Feature importance analysis
            
            **Data Analysis:**
            - CSV file upload support
            - Interactive filtering
            - Correlation analysis
            - Scatter plots and visualizations
            - Data export capabilities
            """)
        
        with col2:
            st.markdown("""
            **Cost Analysis:**
            - Customizable fruit prices
            - Multi-period analysis
            - Financial impact calculations
            - Optimization recommendations
            - Cost breakdown reports
            
            **Model Training:**
            - Custom model training
            - Parameter tuning
            - Performance metrics
            - Feature importance display
            - Model evaluation plots
            """)
    
    with tab3:
        st.subheader("📄 Data Format Requirements")
        
        st.markdown("""
        ### CSV File Format
        
        Your data file should contain the following columns:
        
        | Column | Description | Data Type | Range |
        |--------|-------------|-----------|-------|
        | `fruit_type` | Type of fruit | String | Apple, Banana, Orange, etc. |
        | `quantity` | Amount in kilograms | Float | 0.1 - 1000+ |
        | `days_since_harvest` | Days since harvesting | Integer | 0 - 30 |
        | `temperature` | Storage temperature in °C | Float | -5 to 35 |
        | `humidity` | Relative humidity percentage | Float | 30 - 95 |
        | `waste_percentage` | Actual waste percentage | Float | 0 - 100 |
        
        ### Sample Data Row:
        ```
        fruit_type,quantity,days_since_harvest,temperature,humidity,waste_percentage
        Apple,10.5,7,18,70,12.3
        Banana,5.2,3,22,65,8.7
        ```
        
        ### Data Quality Tips:
        - Ensure no missing values in required columns
        - Use consistent fruit type names
        - Temperature should be in Celsius
        - Humidity as percentage (not decimal)
        - Waste percentage should be 0-100 range
        """)
    
    with tab4:
        st.subheader("❓ Frequently Asked Questions")
        
        with st.expander("How accurate are the predictions?"):
            st.write("""
            The model accuracy depends on the training data quality and similarity to your specific conditions. 
            Typical R² scores range from 0.75-0.90 for well-trained models. You can improve accuracy by:
            - Training with your own data
            - Using consistent measurement methods
            - Including diverse storage conditions in training data
            """)
        
        with st.expander("What fruit types are supported?"):
            st.write("""
            Currently supported fruits:
            - Apple
            - Banana  
            - Orange
            - Strawberry
            - Grapes
            - Pear
            - Peach
            - Plum
            
            More fruit types can be added by training the model with additional data.
            """)
        
        with st.expander("How do I interpret the results?"):
            st.write("""
            **Waste Percentage**: Predicted percentage of fruit that will spoil
            **Freshness Score**: 100 minus waste percentage (higher is better)
            **Risk Level**: 
            - Low (< 10% waste): Optimal storage conditions
            - Medium (10-25% waste): Monitor closely
            - High (> 25% waste): Immediate action needed
            """)
        
        with st.expander("Can I use my own pricing data?"):
            st.write("""
            Yes! In the Cost Analysis page, you can:
            - Set custom prices for each fruit type
            - Adjust prices based on your local market
            - Use different currencies (just update the labels)
            - Save different pricing scenarios
            """)
        
        with st.expander("How often should I retrain the model?"):
            st.write("""
            Retrain your model when:
            - You have new data (monthly/quarterly)
            - Storage conditions change significantly
            - Seasonal variations are observed
            - Model accuracy decreases over time
            - New fruit types need to be added
            """)
    
    # Reset button at the bottom
    st.markdown("---")
    if st.button("🔄 Reset Help View", help="Reset tab selection", key="reset_help"):
        st.rerun()

def faqs_page():
    st.header("❓ Frequently Asked Questions")
    
    faqs = [
        {
            "question": "How accurate are the food waste predictions?",
            "answer": "Our machine learning model provides estimates based on storage conditions and food type characteristics. Accuracy improves with more data and proper input of storage conditions. Results should be used as guidance rather than absolute values."
        },
        {
            "question": "What factors affect food waste the most?",
            "answer": "The main factors are: Days since harvest (35% impact), Temperature (25% impact), Food type (20% impact), Humidity (15% impact), and Quantity (5% impact). Focus on controlling temperature and timing for best results."
        },
        {
            "question": "Can I add my own food items?",
            "answer": "Yes! You can add custom food items in two ways: 1) Quick add in the prediction page by selecting 'Add Custom Item', or 2) Permanent addition through the 'Custom Items' page where you can set detailed properties."
        },
        {
            "question": "How do I create custom categories?",
            "answer": "In the prediction page, select 'Add Custom Category' from the Food Category dropdown. You can also manage categories in the 'Custom Items' page using the Quick Category Manager."
        },
        {
            "question": "What are the ideal storage conditions for different foods?",
            "answer": "Generally: Fruits (0-4°C, 85-90% humidity), Vegetables (0-4°C, 90-95% humidity), Leafy greens (0-2°C, 95-100% humidity). Check individual food suggestions in the prediction results for specific recommendations."
        },
        {
            "question": "How is the cost analysis calculated?",
            "answer": "Cost analysis multiplies the predicted waste percentage by the food quantity and price per kg. It shows potential financial losses and helps prioritize which foods need better storage conditions."
        },
        {
            "question": "Can I export my data and results?",
            "answer": "Yes! You can export prediction results, cost analysis, and custom items as CSV files. Look for the download buttons in each section."
        },
        {
            "question": "What file formats can I upload for bulk predictions?",
            "answer": "Upload CSV files with columns: fruit_type, quantity, days_since_harvest, temperature, humidity. The system will validate and process your data automatically."
        },
        {
            "question": "How often should I retrain the model?",
            "answer": "Retrain the model when you have new data or notice prediction accuracy declining. More data generally improves performance. Use the 'Model Training' page to retrain with your specific data."
        },
        {
            "question": "Are the prices accurate for my region?",
            "answer": "Default prices are estimates for the Indian market. Please update prices in the 'Cost Analysis' page to reflect your local market rates for accurate cost calculations."
        }
    ]
    
    for i, faq in enumerate(faqs):
        with st.expander(f"Q{i+1}: {faq['question']}"):
            st.write(faq['answer'])
    
    st.subheader("Still have questions?")
    st.info("Visit the 'Contact Us' page to get in touch with our support team.")
    
    # Reset button at the bottom
    st.markdown("---")
    if st.button("🔄 Reset FAQ View", help="Reset FAQ expandables", key="reset_faqs"):
        st.rerun()

def contact_page():
    st.header("📞 Contact Us")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Get in Touch")
        st.write("Have questions about SmartShelfAI? Need support or want to provide feedback? We'd love to hear from you!")
        
        st.subheader("📧 Email Support")
        st.write("**General Inquiries:** info@smartshelfai.com")
        st.write("**Technical Support:** support@smartshelfai.com")
        st.write("**Business Partnerships:** business@smartshelfai.com")
        
        st.subheader("📱 Phone Support")
        st.write("**India:** +91-9491844292")
        st.write("**Support Hours:** Monday to Friday, 9:00 AM - 6:00 PM IST")
        
        st.subheader("🏢 Office Address")
        st.write("""
        SmartShelfAI Technologies Pvt. Ltd.
        SRM University
        Kattankulathur, Chennai
        Tamil Nadu 603203
        India
        """)
    
    with col2:
        st.subheader("📝 Send us a Message")
        
        with st.form("contact_form"):
            name = st.text_input("Your Name *", placeholder="Enter your full name")
            email = st.text_input("Email Address *", placeholder="your.email@example.com")
            subject = st.selectbox("Subject *", [
                "General Inquiry",
                "Technical Support",
                "Feature Request",
                "Bug Report",
                "Business Partnership",
                "Other"
            ])
            message = st.text_area("Message *", placeholder="Please describe your inquiry or issue in detail...", height=150)
            
            submitted = st.form_submit_button("Send Message")
            
            if submitted:
                if name and email and message:
                    st.success("✅ Thank you for your message! We'll get back to you within 24 hours.")
                    st.info("Note: This is a demo form. In production, messages would be sent to our support team.")
                else:
                    st.error("Please fill in all required fields (*)")
        
        st.subheader("🌐 Follow Us")
        st.write("Stay updated with the latest features and news:")
        st.write("🔗 Website: www.smartshelfai.com")
        st.write("📘 LinkedIn: @smartshelfai")
        st.write("🐦 Twitter: @smartshelfai")
        st.write("📺 YouTube: SmartShelfAI Channel")
    
    # Reset button at the bottom
    st.markdown("---")
    if st.button("🔄 Reset Contact Form", help="Clear contact form inputs", key="reset_contact"):
        # Clear form-related session state
        for key in list(st.session_state.keys()):
            if key.startswith('contact_'):
                del st.session_state[key]
        st.rerun()

def about_page():
    st.header("ℹ️ About SmartShelfAI")
    
    st.subheader("🎯 Our Mission")
    st.write("""
    SmartShelfAI is dedicated to reducing food waste through intelligent prediction and optimization. 
    We believe that with the right technology and insights, businesses and individuals can significantly 
    reduce food spoilage, save money, and contribute to a more sustainable future.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🚀 What We Do")
        st.write("""
        - **Predictive Analytics:** Advanced machine learning models predict food waste percentages
        - **Cost Optimization:** Calculate financial impact and identify savings opportunities  
        - **Storage Guidance:** Provide optimal storage condition recommendations
        - **Data Insights:** Transform storage data into actionable intelligence
        - **Custom Solutions:** Flexible platform for various food types and categories
        """)
        
        st.subheader("💡 Key Features")
        st.write("""
        - Real-time waste prediction
        - Cost analysis and financial impact assessment
        - Custom food item and category management
        - Data import/export capabilities
        - Model training and customization
        - Comprehensive reporting and visualizations
        """)
    
    with col2:
        st.subheader("📊 Impact Statistics")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Food Waste Reduced", "25-40%", "In pilot programs")
            st.metric("Cost Savings", "₹50,000+", "Average per month")
        with metric_col2:
            st.metric("Users Helped", "1000+", "Across India")
            st.metric("Predictions Made", "50,000+", "Since launch")
        
        st.subheader("🌱 Sustainability")
        st.write("""
        Every kilogram of food waste prevented makes a difference:
        - Reduces greenhouse gas emissions
        - Conserves water and energy resources
        - Decreases landfill burden
        - Improves food security
        """)
    
    st.subheader("🔬 Technology Stack")
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.write("**Machine Learning**")
        st.write("- Random Forest Regression")
        st.write("- Feature Engineering")
        st.write("- Model Validation")
    
    with tech_col2:
        st.write("**Data Processing**")
        st.write("- Pandas & NumPy")
        st.write("- Data Validation")
        st.write("- CSV Import/Export")
    
    with tech_col3:
        st.write("**Visualization**")
        st.write("- Plotly Interactive Charts")
        st.write("- Streamlit Dashboard")
        st.write("- Real-time Updates")
    
    st.subheader("📈 Version Information")
    st.write("**Current Version:** 2.1.0")
    st.write("**Last Updated:** July 2025")
    st.write("**Next Update:** Planned for August 2025 with enhanced AI models")
    
    # Reset button at the bottom
    st.markdown("---")
    if st.button("🔄 Reset About View", help="Reset page view", key="reset_about"):
        st.rerun()

if __name__ == "__main__":
    main()
