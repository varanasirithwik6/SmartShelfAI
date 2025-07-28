import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Visualizer:
    def __init__(self):
        self.color_palette = {
            'primary': '#FF6B6B',
            'secondary': '#4ECDC4',
            'success': '#45B7D1',
            'warning': '#FFA07A',
            'danger': '#FF6B6B',
            'info': '#74B9FF'
        }
    
    def create_waste_gauge(self, waste_percentage):
        """Create a gauge chart for waste percentage"""
        
        # Determine color based on waste level
        if waste_percentage < 10:
            color = "green"
        elif waste_percentage < 25:
            color = "orange"
        else:
            color = "red"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = waste_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Waste Percentage"},
            delta = {'reference': 15, 'suffix': "%"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 10], 'color': "lightgreen"},
                    {'range': [10, 25], 'color': "yellow"},
                    {'range': [25, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            font={'color': "darkblue", 'family': "Arial"},
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        
        return fig
    
    def create_feature_importance_chart(self, importance_dict):
        """Create horizontal bar chart for feature importance"""
        
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        
        # Sort by importance
        sorted_pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        features_sorted, importances_sorted = zip(*sorted_pairs)
        
        fig = go.Figure(go.Bar(
            y=features_sorted,
            x=importances_sorted,
            orientation='h',
            marker_color=self.color_palette['primary'],
            text=[f"{imp:.3f}" for imp in importances_sorted],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=400,
            margin=dict(l=150),
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        
        return fig
    
    def create_historical_scatter(self, data):
        """Create scatter plot for historical waste data"""
        
        fig = px.scatter(
            data,
            x='days_since_harvest',
            y='waste_percentage',
            color='fruit_type',
            size='quantity',
            hover_data=['temperature', 'humidity'],
            title="Historical Waste Patterns",
            labels={
                'days_since_harvest': 'Days Since Harvest',
                'waste_percentage': 'Waste Percentage (%)',
                'fruit_type': 'Fruit Type',
                'quantity': 'Quantity (kg)'
            }
        )
        
        fig.update_layout(
            height=500,
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_prediction_accuracy_plot(self, y_true, y_pred):
        """Create scatter plot showing prediction accuracy"""
        
        # Create perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(
                color=self.color_palette['primary'],
                size=8,
                opacity=0.7
            ),
            text=[f"True: {t:.1f}, Pred: {p:.1f}" for t, p in zip(y_true, y_pred)],
            hovertemplate="<b>Actual:</b> %{x:.1f}%<br><b>Predicted:</b> %{y:.1f}%<extra></extra>"
        ))
        
        # Add perfect prediction line
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title="Prediction Accuracy",
            xaxis_title="Actual Waste Percentage (%)",
            yaxis_title="Predicted Waste Percentage (%)",
            height=500,
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=True
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix):
        """Create correlation heatmap"""
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix"
        )
        
        fig.update_layout(
            height=500,
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        
        return fig
    
    def create_waste_distribution_by_fruit(self, data):
        """Create box plot showing waste distribution by fruit type"""
        
        fig = px.box(
            data,
            x='fruit_type',
            y='waste_percentage',
            title="Waste Distribution by Fruit Type",
            labels={
                'fruit_type': 'Fruit Type',
                'waste_percentage': 'Waste Percentage (%)'
            }
        )
        
        fig.update_layout(
            height=500,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_temperature_impact_plot(self, data):
        """Create scatter plot showing temperature impact on waste"""
        
        fig = px.scatter(
            data,
            x='temperature',
            y='waste_percentage',
            color='fruit_type',
            size='days_since_harvest',
            title="Temperature Impact on Waste Percentage",
            labels={
                'temperature': 'Storage Temperature (°C)',
                'waste_percentage': 'Waste Percentage (%)',
                'days_since_harvest': 'Days Since Harvest'
            }
        )
        
        fig.update_layout(
            height=500,
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        
        return fig
    
    def create_humidity_impact_plot(self, data):
        """Create scatter plot showing humidity impact on waste"""
        
        fig = px.scatter(
            data,
            x='humidity',
            y='waste_percentage',
            color='fruit_type',
            size='quantity',
            title="Humidity Impact on Waste Percentage",
            labels={
                'humidity': 'Humidity (%)',
                'waste_percentage': 'Waste Percentage (%)',
                'quantity': 'Quantity (kg)'
            }
        )
        
        fig.update_layout(
            height=500,
            paper_bgcolor="white",
            plot_bgcolor="white"
        )
        
        return fig
    
    def create_time_series_plot(self, data):
        """Create time series plot for waste trends"""
        
        if 'date' not in data.columns:
            # Generate sample dates if not available
            data = data.copy()
            data['date'] = pd.date_range(start='2024-01-01', periods=len(data), freq='D')
        
        # Group by date and fruit type
        daily_avg = data.groupby(['date', 'fruit_type'])['waste_percentage'].mean().reset_index()
        
        fig = px.line(
            daily_avg,
            x='date',
            y='waste_percentage',
            color='fruit_type',
            title="Waste Percentage Trends Over Time",
            labels={
                'date': 'Date',
                'waste_percentage': 'Average Waste Percentage (%)',
                'fruit_type': 'Fruit Type'
            }
        )
        
        fig.update_layout(
            height=500,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_cost_impact_chart(self, cost_data):
        """Create chart showing cost impact by fruit type"""
        
        fig = go.Figure()
        
        # Add bars for waste cost
        fig.add_trace(go.Bar(
            name='Waste Cost',
            x=cost_data['Fruit'],
            y=cost_data['Waste Cost ($)'],
            marker_color=self.color_palette['danger'],
            text=cost_data['Waste Cost ($)'].apply(lambda x: f"${x:.2f}"),
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Financial Impact of Food Waste by Fruit Type",
            xaxis_title="Fruit Type",
            yaxis_title="Waste Cost ($)",
            height=500,
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False
        )
        
        return fig
