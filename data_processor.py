import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self):
        self.required_columns = [
            'fruit_type', 'quantity', 'days_since_harvest', 
            'temperature', 'humidity', 'waste_percentage'
        ]
    
    def validate_data(self, df):
        """Validate uploaded data format and content"""
        errors = []
        warnings = []
        
        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")
        
        if errors:
            return False, errors, warnings
        
        # Check data types and ranges
        try:
            # Numeric columns validation
            numeric_cols = ['quantity', 'days_since_harvest', 'temperature', 'humidity', 'waste_percentage']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Check for NaN values after conversion
                    nan_count = df[col].isna().sum()
                    if nan_count > 0:
                        warnings.append(f"{col}: {nan_count} non-numeric values converted to NaN")
            
            # Range validations
            if 'quantity' in df.columns:
                invalid_qty = ((df['quantity'] < 0) | (df['quantity'] > 10000)).sum()
                if invalid_qty > 0:
                    warnings.append(f"quantity: {invalid_qty} values outside normal range (0-10000 kg)")
            
            if 'days_since_harvest' in df.columns:
                invalid_days = ((df['days_since_harvest'] < 0) | (df['days_since_harvest'] > 60)).sum()
                if invalid_days > 0:
                    warnings.append(f"days_since_harvest: {invalid_days} values outside normal range (0-60 days)")
            
            if 'temperature' in df.columns:
                invalid_temp = ((df['temperature'] < -20) | (df['temperature'] > 50)).sum()
                if invalid_temp > 0:
                    warnings.append(f"temperature: {invalid_temp} values outside normal range (-20 to 50°C)")
            
            if 'humidity' in df.columns:
                invalid_humid = ((df['humidity'] < 0) | (df['humidity'] > 100)).sum()
                if invalid_humid > 0:
                    warnings.append(f"humidity: {invalid_humid} values outside normal range (0-100%)")
            
            if 'waste_percentage' in df.columns:
                invalid_waste = ((df['waste_percentage'] < 0) | (df['waste_percentage'] > 100)).sum()
                if invalid_waste > 0:
                    warnings.append(f"waste_percentage: {invalid_waste} values outside normal range (0-100%)")
            
            # Food type validation
            if 'fruit_type' in df.columns:
                valid_foods = [
                    'Apple', 'Banana', 'Orange', 'Strawberry', 'Grapes', 
                    'Tomato', 'Carrot', 'Potato', 'Lettuce', 'Spinach', 'Lemon'
                ]
                invalid_foods = df[~df['fruit_type'].isin(valid_foods)]['fruit_type'].unique()
                if len(invalid_foods) > 0:
                    warnings.append(f"Unknown food types found: {', '.join(invalid_foods[:5])}... (will be mapped to similar items)")
            
        except Exception as e:
            errors.append(f"Data validation error: {str(e)}")
        
        return len(errors) == 0, errors, warnings
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        df_clean = df.copy()
        
        # Remove rows with critical missing values
        df_clean = df_clean.dropna(subset=['fruit_type', 'waste_percentage'])
        
        # Fill missing numeric values with median
        numeric_cols = ['quantity', 'days_since_harvest', 'temperature', 'humidity']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Clamp values to reasonable ranges
        if 'quantity' in df_clean.columns:
            df_clean['quantity'] = df_clean['quantity'].clip(0.1, 10000)
        
        if 'days_since_harvest' in df_clean.columns:
            df_clean['days_since_harvest'] = df_clean['days_since_harvest'].clip(0, 60)
        
        if 'temperature' in df_clean.columns:
            df_clean['temperature'] = df_clean['temperature'].clip(-20, 50)
        
        if 'humidity' in df_clean.columns:
            df_clean['humidity'] = df_clean['humidity'].clip(0, 100)
        
        if 'waste_percentage' in df_clean.columns:
            df_clean['waste_percentage'] = df_clean['waste_percentage'].clip(0, 100)
        
        # Standardize food type names
        valid_foods = [
            'Apple', 'Banana', 'Orange', 'Strawberry', 'Grapes', 
            'Tomato', 'Carrot', 'Potato', 'Lettuce', 'Spinach', 'Lemon'
        ]
        
        def map_to_valid_food(food_name):
            if food_name in valid_foods:
                return food_name
            
            # Smart mapping for similar foods
            food_lower = food_name.lower()
            if any(word in food_lower for word in ['berry', 'cherry']):
                return 'Strawberry'
            elif any(word in food_lower for word in ['lettuce', 'greens', 'herbs', 'kale', 'spinach']):
                return 'Lettuce' 
            elif any(word in food_lower for word in ['potato', 'root', 'tuber']):
                return 'Potato'
            elif any(word in food_lower for word in ['citrus', 'lime', 'grapefruit']):
                return 'Lemon'
            elif any(word in food_lower for word in ['tomato', 'pepper']):
                return 'Tomato'
            else:
                return 'Apple'
        
        df_clean['fruit_type'] = df_clean['fruit_type'].apply(map_to_valid_food)
        
        return df_clean
    
    def get_data_summary(self, df):
        """Generate summary statistics for the data"""
        summary = {}
        
        # Basic info
        summary['total_records'] = len(df)
        summary['date_range'] = {
            'start': datetime.now() - timedelta(days=30),
            'end': datetime.now()
        }
        
        # Fruit distribution
        if 'fruit_type' in df.columns:
            summary['fruit_distribution'] = df['fruit_type'].value_counts().to_dict()
        
        # Numeric summaries
        numeric_cols = ['quantity', 'days_since_harvest', 'temperature', 'humidity', 'waste_percentage']
        summary['numeric_stats'] = {}
        
        for col in numeric_cols:
            if col in df.columns:
                summary['numeric_stats'][col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'missing': df[col].isna().sum()
                }
        
        # Correlations
        if len(numeric_cols) > 1:
            available_cols = [col for col in numeric_cols if col in df.columns]
            if len(available_cols) > 1:
                summary['correlations'] = df[available_cols].corr().to_dict()
        
        return summary
    
    def filter_data(self, df, filters):
        """Apply filters to the data"""
        filtered_df = df.copy()
        
        # Fruit type filter
        if 'fruit_types' in filters and filters['fruit_types']:
            filtered_df = filtered_df[filtered_df['fruit_type'].isin(filters['fruit_types'])]
        
        # Date range filter (if date column exists)
        if 'date_range' in filters and 'date' in filtered_df.columns:
            start_date, end_date = filters['date_range']
            filtered_df = filtered_df[
                (filtered_df['date'] >= start_date) & 
                (filtered_df['date'] <= end_date)
            ]
        
        # Numeric range filters
        numeric_filters = [
            ('days_since_harvest', 'days_range'),
            ('temperature', 'temp_range'),
            ('humidity', 'humidity_range'),
            ('waste_percentage', 'waste_range')
        ]
        
        for col, filter_key in numeric_filters:
            if filter_key in filters and col in filtered_df.columns:
                min_val, max_val = filters[filter_key]
                filtered_df = filtered_df[
                    (filtered_df[col] >= min_val) & 
                    (filtered_df[col] <= max_val)
                ]
        
        return filtered_df
    
    def export_data(self, df, format='csv'):
        """Export data in specified format"""
        if format.lower() == 'csv':
            return df.to_csv(index=False)
        elif format.lower() == 'json':
            return df.to_json(orient='records', indent=2)
        elif format.lower() == 'excel':
            # For Excel export, we'll return the dataframe
            # The actual Excel file creation should be handled by the calling function
            return df
        else:
            raise ValueError(f"Unsupported export format: {format}")
