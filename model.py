import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class FoodWastePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.feature_names = ['fruit_type_encoded', 'quantity', 'days_since_harvest', 'temperature', 'humidity']
        self.is_trained = False
        self.feature_importance_ = None
        
        # Initialize with basic training if no model exists
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model with basic training data if no trained model exists"""
        try:
            # Try to load existing model
            self.load_model()
        except:
            # Train with minimal sample data if no model exists
            from utils import generate_sample_data
            sample_data = generate_sample_data(100)
            self.train(sample_data)
    
    def update_parameters(self, n_estimators=100, max_depth=10, min_samples_split=5):
        """Update model parameters"""
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        self.is_trained = False
    
    def prepare_features(self, data):
        """Prepare features for training or prediction"""
        if isinstance(data, dict):
            # Single prediction
            df = pd.DataFrame([data])
        else:
            # Multiple samples
            df = data.copy()
        
        # Encode food types
        if not hasattr(self, 'fruit_types_'):
            # First time encoding - fit the encoder with basic food items
            unique_foods = [
                'Apple', 'Banana', 'Orange', 'Strawberry', 'Grapes', 
                'Tomato', 'Carrot', 'Potato', 'Lettuce', 'Spinach', 'Lemon'
            ]
            self.label_encoder.fit(unique_foods)
            self.fruit_types_ = unique_foods
        
        # Handle unknown food types by finding the closest match or using Apple as fallback
        def map_food_type(food_name):
            if food_name in self.fruit_types_:
                return food_name
            
            # Try to find similar food types (simple matching)
            food_lower = food_name.lower()
            
            # Mapping rules for common categories
            if any(word in food_lower for word in ['berry', 'cherry', 'grape']):
                return 'Strawberry'  # High perishability
            elif any(word in food_lower for word in ['lettuce', 'spinach', 'greens', 'herbs', 'kale']):
                return 'Lettuce'  # High perishability
            elif any(word in food_lower for word in ['potato', 'root', 'tuber']):
                return 'Potato'  # Low perishability
            elif any(word in food_lower for word in ['citrus', 'lime', 'grapefruit']):
                return 'Lemon'  # Medium-low perishability
            elif any(word in food_lower for word in ['tomato', 'pepper', 'cucumber']):
                return 'Tomato'  # Medium perishability
            elif any(word in food_lower for word in ['carrot', 'vegetable']):
                return 'Carrot'  # Medium perishability
            else:
                return 'Apple'  # Default fallback
        
        df['fruit_type'] = df['fruit_type'].apply(map_food_type)
        
        df['fruit_type_encoded'] = self.label_encoder.transform(df['fruit_type'])
        
        # Select and order features
        features = df[self.feature_names]
        return features
    
    def train(self, training_data):
        """Train the model with provided data"""
        try:
            # Prepare features and target
            X = self.prepare_features(training_data)
            y = training_data['waste_percentage']
            
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train the model
            self.model.fit(X_train, y_train)
            self.is_trained = True
            self.feature_importance_ = self.model.feature_importances_
            
            # Calculate metrics
            y_pred = self.model.predict(X_test)
            
            metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'y_true': list(y_test),
                'predictions': y_pred.tolist()
            }
            
            # Save the model
            self.save_model()
            
            return metrics
            
        except Exception as e:
            raise Exception(f"Training failed: {str(e)}")
    
    def predict(self, features):
        """Make prediction for given features"""
        if not self.is_trained:
            # Use basic heuristic if model not trained
            return self._heuristic_prediction(features)
        
        try:
            X = self.prepare_features(features)
            prediction = self.model.predict(X)[0]
            
            # Ensure prediction is within reasonable bounds
            prediction = max(0, min(100, prediction))
            
            return round(prediction, 1)
            
        except Exception as e:
            # Fallback to heuristic
            return self._heuristic_prediction(features)
    
    def _heuristic_prediction(self, features):
        """Fallback heuristic prediction when model is not available"""
        # Basic waste prediction based on rules
        base_waste = 5.0  # Base waste percentage
        
        # Days since harvest impact
        days_impact = features['days_since_harvest'] * 1.5
        
        # Temperature impact (optimal around 4°C for most fruits)
        temp_optimal = 4
        temp_impact = abs(features['temperature'] - temp_optimal) * 0.5
        
        # Humidity impact (optimal around 85-90% for most fruits)
        humidity_optimal = 87.5
        humidity_impact = abs(features['humidity'] - humidity_optimal) * 0.1
        
        # Food type impact (different spoilage rates)
        food_multipliers = {
            # Highly perishable
            'Strawberry': 2.0, 'Lettuce': 1.8, 'Spinach': 1.9,
            # Moderately perishable
            'Banana': 1.5, 'Tomato': 1.4,
            # Less perishable
            'Grapes': 1.2, 'Apple': 1.0, 'Orange': 0.9, 'Carrot': 0.8, 'Potato': 0.6,
            # Citrus (longer lasting)
            'Lemon': 0.7
        }
        
        food_impact = food_multipliers.get(features['fruit_type'], 1.0)
        
        # Calculate total waste
        waste_percentage = (base_waste + days_impact + temp_impact + humidity_impact) * food_impact
        
        # Ensure reasonable bounds
        waste_percentage = max(0, min(80, waste_percentage))
        
        return round(waste_percentage, 1)
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.feature_importance_ is not None:
            feature_names = ['Food Type', 'Quantity', 'Days Since Harvest', 'Temperature', 'Humidity']
            return dict(zip(feature_names, self.feature_importance_))
        else:
            # Return default importance if model not trained
            return {
                'Days Since Harvest': 0.35,
                'Temperature': 0.25,
                'Food Type': 0.20,
                'Humidity': 0.15,
                'Quantity': 0.05
            }
    
    def save_model(self):
        """Save the trained model"""
        try:
            model_data = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'fruit_types': self.fruit_types_,
                'is_trained': self.is_trained,
                'feature_importance': self.feature_importance_
            }
            joblib.dump(model_data, 'food_waste_model.pkl')
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
    
    def load_model(self):
        """Load a previously trained model"""
        try:
            model_data = joblib.load('food_waste_model.pkl')
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            self.fruit_types_ = model_data['fruit_types']
            self.is_trained = model_data['is_trained']
            self.feature_importance_ = model_data.get('feature_importance')
            print("Model loaded successfully")
        except Exception as e:
            raise Exception(f"Could not load model: {e}")
