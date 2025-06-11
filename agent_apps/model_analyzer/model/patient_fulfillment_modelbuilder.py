import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class PatientFulfillmentModelBuilder:
    def __init__(self, n_samples=1000, random_seed=42):
        self.n_samples = n_samples
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        self.df = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.pipeline = None
        self.X_test_transformed = None

    def generate_synthetic_data(self):
        """Generate synthetic patient fulfillment dataset."""
        sp_names = ["Caremark", "Briova", "Senderra", "Accredo", "Wegmans", "Blue Sky", "Kroger", "United Health"]
        payer_plans = ["Medicare", "Medicaid", "CVS Caremark", "BCBS", "United Health"]
        genders = ["Male", "Female"]
        therapy_types = ["Subcutaneous", "Other"]

        df = pd.DataFrame({
            'sp_name': np.random.choice(sp_names, self.n_samples),
            'payer_plan': np.random.choice(payer_plans, self.n_samples),
            'plan_switch_last_year': np.random.randint(0, 2, self.n_samples),
            'out_of_pocket_cost': np.random.normal(loc=100, scale=30, size=self.n_samples).clip(min=0),
            'pa_approval_ratio': np.random.uniform(0.5, 1.0, self.n_samples),
            'hcp_biologics_experience': np.random.randint(0, 2, self.n_samples),
            'hcp_subq_experience': np.random.randint(0, 2, self.n_samples),
            'hcp_age': np.random.randint(30, 71, self.n_samples),
            'hcp_gender': np.random.choice(genders, self.n_samples),
            'practice_size': np.random.randint(1, 51, self.n_samples),
            'patient_age': np.random.randint(18, 91, self.n_samples),
            'patient_gender': np.random.choice(genders, self.n_samples),
            'patient_married': np.random.randint(0, 2, self.n_samples),
            'time_since_diagnosis_months': np.random.randint(1, 121, self.n_samples),
            'enrolled_through_hub': np.random.randint(0, 2, self.n_samples),
            'therapy_type': np.random.choice(therapy_types, self.n_samples)
        })

        # Fulfillment probability heuristic
        base_prob = 0.6
        prob = (
            base_prob
            + 0.1 * df['hcp_biologics_experience']
            + 0.07 * df['enrolled_through_hub']
            + 0.08 * df['hcp_subq_experience']
            - 0.05 * df['plan_switch_last_year']
            - 0.08 * (df['out_of_pocket_cost'] > 200).astype(int)
        ).clip(0, 1)

        df['is_fulfilled_28d'] = np.random.binomial(1, prob)

        self.df = df
        print("✅ Synthetic data generated.")

    def preprocess_data(self):
        """One-hot encode categorical variables and split the dataset."""
        df_encoded = pd.get_dummies(
            self.df,
            columns=[
                'sp_name', 'payer_plan', 'hcp_gender', 'patient_gender', 'therapy_type'
            ],
            drop_first=True
        )

        X = df_encoded.drop('is_fulfilled_28d', axis=1)
        y = df_encoded['is_fulfilled_28d']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed
        )

        self.feature_names = list(X.columns)
        self.X_test_transformed = self.X_test.copy()

        print("✅ Data preprocessed and split.")

    def train_model(self):
        """Train RandomForest model."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_seed)
        self.model.fit(self.X_train, self.y_train)
        print("✅ Model trained.")

    def save_pickle(self, output_file='patient_fulfillment_model.pkl'):
        """Save model, test data, and metadata into a single pickle file."""
        model_data = {
            'model': self.model,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'X_test_transformed': self.X_test_transformed,
            'feature_names': self.feature_names
        }

        with open(output_file, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Model and data saved to '{output_file}'.")

    def build_and_save_model(self, output_file='patient_fulfillment_model.pkl'):
        """End-to-end pipeline."""
        self.generate_synthetic_data()
        self.preprocess_data()
        self.train_model()
        self.save_pickle(output_file)
