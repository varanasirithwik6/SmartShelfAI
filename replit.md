# SmartShelfAI - Food Waste Prediction System

## Overview

SmartShelfAI is a Streamlit-based web application that uses machine learning to predict food waste percentages for various fruits. The system helps optimize storage conditions and reduce spoilage by analyzing factors like fruit type, quantity, storage duration, temperature, and humidity. The application provides an intuitive interface for predictions, data analysis, model training, and cost analysis.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid prototyping and deployment
- **Layout**: Wide layout with expandable sidebar navigation
- **Caching**: Streamlit's `@st.cache_resource` decorator for component initialization
- **Navigation**: Multi-page application with sidebar selection menu

### Backend Architecture
- **Core Components**: Modular architecture with separate classes for different functionalities
  - `FoodWastePredictor`: Machine learning model management
  - `DataProcessor`: Data validation and preprocessing
  - `Visualizer`: Chart and visualization generation
- **Model**: RandomForestRegressor from scikit-learn for waste prediction
- **Data Processing**: Pandas for data manipulation and NumPy for numerical operations

### Machine Learning Pipeline
- **Algorithm**: Random Forest Regression chosen for its robustness and interpretability
- **Features**: Fruit type (encoded), quantity, days since harvest, temperature, humidity
- **Target**: Waste percentage prediction
- **Preprocessing**: Label encoding for categorical variables, data validation and cleaning

## Key Components

### 1. Model Management (`model.py`)
- **Purpose**: Handles ML model training, prediction, and persistence
- **Key Features**:
  - Automatic model initialization with sample data
  - Parameter tuning capabilities
  - Model saving/loading functionality
  - Feature importance analysis

### 2. Data Processing (`data_processor.py`)
- **Purpose**: Validates and preprocesses input data
- **Validation Rules**:
  - Required columns enforcement
  - Data type validation and conversion
  - Range checking for realistic values
  - Error and warning reporting system

### 3. Visualizations (`visualizations.py`)
- **Purpose**: Creates interactive charts and graphs using Plotly
- **Chart Types**:
  - Gauge charts for waste percentage display
  - Color-coded indicators based on waste levels
  - Interactive plotting capabilities

### 4. Utilities (`utils.py`)
- **Purpose**: Provide helper functions for data generation and processing
- **Key Functions**:
  - Sample data generation with fruit-specific parameters
  - Data export functionality
  - Cost estimation calculations

### 5. Main Application (`app.py`)
- **Purpose**: Orchestrates the entire application
- **Page Structure**:
  - Prediction interface
  - Data analysis dashboard
  - Model training interface
  - Cost analysis tools
  - Help and guidance

## Data Flow

1. **Input Stage**: Users provide fruit data through file upload or manual entry
2. **Validation**: DataProcessor validates data format and content
3. **Processing**: Data is preprocessed and features are prepared
4. **Prediction**: FoodWastePredictor generates waste percentage predictions
5. **Visualization**: Results are displayed through interactive charts and metrics
6. **Export**: Processed results can be exported for further analysis

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and tools
- **Plotly**: Interactive visualization library

### Machine Learning Stack
- **RandomForestRegressor**: Primary prediction model
- **LabelEncoder**: Categorical variable encoding
- **Train-test split**: Model validation
- **Metrics**: R², MSE, MAE for performance evaluation

### Utility Libraries
- **Base64**: Data encoding for file operations
- **IO**: Input/output operations
- **Datetime**: Date and time handling
- **Joblib**: Model serialization

## Deployment Strategy

### Streamlit Deployment
- **Platform**: Designed for Streamlit Cloud or similar platforms
- **Configuration**: Uses `st.set_page_config()` for optimal display
- **Caching**: Implements Streamlit caching for performance optimization
- **Resource Management**: Components are cached to reduce initialization overhead

### Model Persistence
- **Storage**: Models are saved/loaded using joblib for persistence
- **Initialization**: Automatic fallback to sample data training if no saved model exists
- **Updates**: Dynamic model retraining capabilities through the interface

### Performance Considerations
- **Caching Strategy**: Resource-intensive components are cached
- **Data Validation**: Efficient validation with clear error reporting
- **Memory Management**: Streamlit's built-in resource management
- **Scalability**: Modular design allows for easy component replacement and scaling

The architecture prioritizes simplicity, maintainability, and user experience while providing robust machine learning capabilities for food waste prediction.