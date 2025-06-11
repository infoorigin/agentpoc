import pickle

class ModelLoader:
    def __init__(self, pickle_file_path):
        self.pickle_file_path = pickle_file_path
        self.model = None
        self.feature_names = None
        self.X_test = None
        self.y_test = None

    def load_pickle(self):
        """Load the pickle file containing the model and metadata."""
        with open(self.pickle_file_path, 'rb') as f:
            data = pickle.load(f)

        self.model = data.get('model', None)
        self.feature_names = data.get('feature_names', None)
        self.X_test = data.get('X_test', None)
        self.y_test = data.get('y_test', None)

        if self.model is None or self.feature_names is None:
            raise ValueError("Pickle file missing required fields: 'model' or 'feature_names'.")

        print(f"✅ Pickle loaded successfully from '{self.pickle_file_path}'.")

    def get_model_data(self):
        """Return loaded model, feature names, and test data."""
        return self.model, self.feature_names, self.X_test, self.y_test
