import pandas as pd
import numpy as np
import base64
import io
from datetime import datetime, timedelta

def generate_sample_data(n_samples=200):
    """Generate sample food waste data for demonstration"""
    np.random.seed(42)  # For reproducible results
    
    foods = [
        'Apple', 'Banana', 'Orange', 'Strawberry', 'Grapes', 
        'Tomato', 'Carrot', 'Potato', 'Lettuce', 'Spinach', 'Lemon'
    ]
    
    # Generate random data
    data = []
    
    for _ in range(n_samples):
        food_item = np.random.choice(foods)
        
        # Base parameters with food-specific variations
        food_params = {
            # Highly perishable items
            'Strawberry': (15, 2.0, 2.5), 'Lettuce': (14, 1.8, 2.3), 'Spinach': (15, 1.9, 2.4),
            # Moderately perishable
            'Banana': (12, 1.8, 2.0), 'Tomato': (13, 1.4, 2.1),
            # Less perishable
            'Grapes': (10, 1.4, 1.8), 'Apple': (8, 1.0, 1.5), 'Orange': (7, 0.8, 1.3), 
            'Carrot': (6, 0.8, 1.2), 'Potato': (4, 0.6, 1.0), 'Lemon': (5, 0.7, 1.1)
        }
        
        base_waste, temp_sensitivity, days_multiplier = food_params.get(food_item, (8, 1.0, 1.5))
        
        # Generate realistic parameters
        quantity = np.random.uniform(1, 50)
        days_since_harvest = np.random.randint(0, 21)
        temperature = np.random.normal(20, 8)  # Room temperature with variation
        temperature = max(-5, min(35, temperature))  # Clamp to realistic range
        humidity = np.random.normal(65, 15)
        humidity = max(30, min(95, humidity))  # Clamp to realistic range
        
        # Calculate waste percentage based on realistic factors
        # Days since harvest impact
        days_impact = days_since_harvest * days_multiplier
        
        # Temperature impact (optimal around 4°C for most fruits)
        temp_optimal = 4
        temp_impact = abs(temperature - temp_optimal) * temp_sensitivity * 0.5
        
        # Humidity impact (optimal around 85-90% for most fruits)
        humidity_optimal = 87.5
        humidity_impact = abs(humidity - humidity_optimal) * 0.1
        
        # Random variation
        random_factor = np.random.normal(0, 2)
        
        # Calculate final waste percentage
        waste_percentage = base_waste + days_impact + temp_impact + humidity_impact + random_factor
        waste_percentage = max(0, min(80, waste_percentage))  # Realistic bounds
        
        data.append({
            'fruit_type': food_item,
            'quantity': round(quantity, 1),
            'days_since_harvest': days_since_harvest,
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'waste_percentage': round(waste_percentage, 1)
        })
    
    return pd.DataFrame(data)

def calculate_cost_estimate(waste_percentage, quantity, price_per_kg):
    """Calculate financial cost of food waste"""
    waste_kg = (waste_percentage / 100) * quantity
    waste_cost = waste_kg * price_per_kg
    total_value = quantity * price_per_kg
    
    return {
        'waste_kg': waste_kg,
        'waste_cost': waste_cost,
        'total_value': total_value,
        'loss_percentage': (waste_cost / total_value) * 100 if total_value > 0 else 0
    }

def export_data(data, filename, format='csv'):
    """Export data in various formats"""
    
    if format.lower() == 'csv':
        csv_string = data.to_csv(index=False)
        return csv_string
    
    elif format.lower() == 'json':
        json_string = data.to_json(orient='records', indent=2)
        return json_string
    
    elif format.lower() == 'excel':
        # Create Excel file in memory
        output = io.BytesIO()
        data.to_excel(output, sheet_name='Food_Waste_Data', index=False, engine='openpyxl')
        
        excel_data = output.getvalue()
        return base64.b64encode(excel_data).decode()
    
    else:
        raise ValueError(f"Unsupported format: {format}")

def create_download_link(data, filename, format='csv'):
    """Create a download link for data export"""
    
    if format.lower() == 'csv':
        csv_string = export_data(data, filename, 'csv')
        b64 = base64.b64encode(csv_string.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV File</a>'
        return href
    
    elif format.lower() == 'json':
        json_string = export_data(data, filename, 'json')
        b64 = base64.b64encode(json_string.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="{filename}.json">Download JSON File</a>'
        return href
    
    elif format.lower() == 'excel':
        excel_b64 = export_data(data, filename, 'excel')
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_b64}" download="{filename}.xlsx">Download Excel File</a>'
        return href
    
    else:
        raise ValueError(f"Unsupported format: {format}")

def validate_input_ranges(features):
    """Validate input parameter ranges"""
    errors = []
    
    # Quantity validation
    if features.get('quantity', 0) <= 0:
        errors.append("Quantity must be greater than 0")
    elif features.get('quantity', 0) > 10000:
        errors.append("Quantity seems unreasonably high (>10,000 kg)")
    
    # Days since harvest validation
    if features.get('days_since_harvest', 0) < 0:
        errors.append("Days since harvest cannot be negative")
    elif features.get('days_since_harvest', 0) > 60:
        errors.append("Days since harvest seems too high (>60 days)")
    
    # Temperature validation
    if features.get('temperature', 0) < -20:
        errors.append("Temperature too low (<-20°C)")
    elif features.get('temperature', 0) > 50:
        errors.append("Temperature too high (>50°C)")
    
    # Humidity validation
    if features.get('humidity', 0) < 0:
        errors.append("Humidity cannot be negative")
    elif features.get('humidity', 0) > 100:
        errors.append("Humidity cannot exceed 100%")
    
    return errors

def get_optimization_suggestions(features, waste_percentage):
    """Generate optimization suggestions based on current conditions"""
    suggestions = []
    
    # Temperature suggestions
    temp = features.get('temperature', 20)
    if temp > 25:
        suggestions.append(f"🌡️ Consider lowering storage temperature (currently {temp}°C). Optimal range is 2-6°C for most fruits.")
    elif temp < 0:
        suggestions.append(f"🌡️ Storage temperature is too low ({temp}°C). This may cause freezing damage.")
    
    # Humidity suggestions
    humidity = features.get('humidity', 65)
    if humidity < 60:
        suggestions.append(f"💧 Humidity is low ({humidity}%). Consider increasing to 80-90% for better preservation.")
    elif humidity > 95:
        suggestions.append(f"💧 Humidity is very high ({humidity}%). This may promote mold growth.")
    
    # Days since harvest suggestions
    days = features.get('days_since_harvest', 0)
    if days > 14:
        suggestions.append(f"⏰ Fruit has been stored for {days} days. Consider using soon to minimize waste.")
    elif days > 21:
        suggestions.append(f"⚠️ Fruit is {days} days old. Immediate consumption or processing recommended.")
    
    # Waste level suggestions
    if waste_percentage > 25:
        suggestions.append("🚨 High waste prediction! Review storage conditions immediately.")
    elif waste_percentage > 15:
        suggestions.append("⚠️ Moderate waste risk. Monitor conditions closely.")
    elif waste_percentage < 5:
        suggestions.append("✅ Excellent storage conditions! Current practices are optimal.")
    
    # Food-specific suggestions
    food_type = features.get('fruit_type', '')
    food_suggestions = {
        'Strawberry': "🍓 Highly perishable. Keep at 0-2°C and handle gently.",
        'Raspberry': "🫐 Very delicate. Store at 0-2°C, consume within 2-3 days.",
        'Blueberry': "🫐 Keep refrigerated at 0-4°C. Don't wash until ready to eat.",
        'Banana': "🍌 Ripen quickly. Store at 13-14°C to slow ripening.",
        'Apple': "🍎 Store well. Keep at 0-4°C with high humidity.",
        'Lettuce': "🥬 Keep very cold (0-2°C) and moist to prevent wilting.",
        'Spinach': "🥬 Highly perishable. Store at 0-2°C and use within 3-5 days.",
        'Tomato': "🍅 Store at room temperature until ripe, then refrigerate.",
        'Avocado': "🥑 Ripen at room temperature, then refrigerate when ripe.",
        'Potato': "🥔 Store in cool, dark place. Avoid refrigeration.",
        'Carrot': "🥕 Remove tops and store in refrigerator at high humidity.",
        'Broccoli': "🥦 Keep very cold and moist. Use within 3-5 days."
    }
    
    if food_type in food_suggestions:
        suggestions.append(f"{food_suggestions[food_type]}")
    
    return suggestions[:5]  # Limit to top 5 suggestions

def format_currency(amount, currency='INR'):
    """Format currency amounts"""
    if currency == 'INR':
        return f"₹{amount:,.0f}"
    elif currency == 'USD':
        return f"${amount:.2f}"
    elif currency == 'EUR':
        return f"€{amount:.2f}"
    elif currency == 'GBP':
        return f"£{amount:.2f}"
    else:
        return f"{amount:.2f} {currency}"

def calculate_savings_potential(current_waste_pct, improved_waste_pct, quantity, price_per_kg):
    """Calculate potential savings from waste reduction"""
    current_waste_cost = calculate_cost_estimate(current_waste_pct, quantity, price_per_kg)['waste_cost']
    improved_waste_cost = calculate_cost_estimate(improved_waste_pct, quantity, price_per_kg)['waste_cost']
    
    savings = current_waste_cost - improved_waste_cost
    savings_percentage = (savings / current_waste_cost) * 100 if current_waste_cost > 0 else 0
    
    return {
        'savings_amount': savings,
        'savings_percentage': savings_percentage,
        'current_cost': current_waste_cost,
        'improved_cost': improved_waste_cost
    }
