import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class ChurnPredictor:
    def __init__(self):
        # We use a constrained Random Forest for fast, real-time UI inference
        self.model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        self.encoders = {}
        # Top 6 features highly correlated with churn from your EDA
        self.features = ['tenure', 'MonthlyCharges', 'Contract', 'InternetService', 'TechSupport', 'PaymentMethod']
        
    def train(self, df):
        """Trains the model dynamically on app startup."""
        df_train = df.dropna(subset=['Churn'] + self.features).copy()
        
        # Encode categorical variables
        cat_cols = ['Contract', 'InternetService', 'TechSupport', 'PaymentMethod']
        for col in cat_cols:
            le = LabelEncoder()
            df_train[col] = le.fit_transform(df_train[col].astype(str))
            self.encoders[col] = le
            
        X = df_train[self.features]
        y = df_train['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        self.model.fit(X, y)
        print("--- Machine Learning Model Trained Successfully ---")
        
    def predict(self, customer_dict):
        """Takes a dictionary of customer inputs from the UI and returns a churn probability %"""
        input_df = pd.DataFrame([customer_dict])
        
        # Apply the saved encoders to the UI input
        for col in ['Contract', 'InternetService', 'TechSupport', 'PaymentMethod']:
            if col in self.encoders:
                # Handle potential unseen UI inputs gracefully
                try:
                    input_df[col] = self.encoders[col].transform(input_df[col])
                except ValueError:
                    input_df[col] = 0 
                    
        # Return the probability of class 1 (Churn = Yes)
        proba = self.model.predict_proba(input_df[self.features])[0][1]
        return round(proba * 100, 2)