import shap
import matplotlib.pyplot as plt
import os
import numpy as np
import io
import base64

class SHAPPlotGenerator:
    def __init__(self, model, X_test, output_dir='shap_outputs', random_seed=42):
        self.model = model
        self.X_test = X_test
        self.output_dir = output_dir
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        
        self.explainer = None
        self.shap_values = None

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def calculate_shap_values(self):
        """Calculate SHAP values using TreeExplainer."""
        self.explainer = shap.TreeExplainer(self.model)
        shap_raw = self.explainer.shap_values(self.X_test)

        if isinstance(shap_raw, list):
            # Old API (list of arrays)
            if len(shap_raw) == 2:
                self.shap_values = shap_raw[1]  # Pick class 1
            else:
                raise ValueError("Multi-class classification not handled.")
        else:
            # New API (3D array returned)
            if shap_raw.ndim == 3:
                # (n_samples, n_features, n_classes) → take positive class (churn=1)
                self.shap_values = shap_raw[:, :, 1]  # 🔥 Take SHAP values for class 1
            else:
                self.shap_values = shap_raw

        # Sanity check
        assert self.shap_values.shape == self.X_test.shape, f"Mismatch after fixing! shap={self.shap_values.shape}, x_test={self.X_test.shape}"
        print(f"✅ Corrected SHAP values shape = {self.shap_values.shape}, X_test shape = {self.X_test.shape}")


    def get_summary_plot_base64(self) -> str:
        """
        Generate the SHAP summary plot and return it as a base64-encoded PNG image.

        Returns:
            str: Base64-encoded string representing the summary plot image.
        """
        plt.figure()
        shap.summary_plot(self.shap_values, self.X_test, show=False)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return image_base64

    def save_summary_plot(self):
        """Save global SHAP summary plot."""
        plt.figure()
        shap.summary_plot(self.shap_values, self.X_test, show=False)
        plt.title("Global Feature Importance")
        plt.savefig(os.path.join(self.output_dir, "summary_plot.png"), bbox_inches="tight")
        plt.close()
 

    def save_summary_plot_top_n_features(self, top_n=5):
        """Save global SHAP summary bar plot for top N features only."""
        import numpy as np
        import matplotlib.pyplot as plt
        import shap
        import os

        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]

        # Slice SHAP values and X_test accordingly
        shap_values_top = self.shap_values[:, top_indices]
        X_test_top = self.X_test.iloc[:, top_indices]

        # Plot
        plt.figure()
        shap.summary_plot(shap_values_top, X_test_top, show=False, plot_type="bar")
        plt.title(f"Top {top_n} Feature Importances")
        save_path = os.path.join(self.output_dir, f"summary_bar_top{top_n}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"✅ Summary bar plot (top {top_n}) saved at {save_path}")
    

    def save_bar_plot(self):
        """Save mean absolute SHAP value bar plot."""
        plt.figure()
        shap.summary_plot(self.shap_values, self.X_test, plot_type="bar", show=False)
        plt.title("Mean Absolute Feature Importance")
        plt.savefig(os.path.join(self.output_dir, "feature_importance_bar.png"), bbox_inches="tight")
        plt.close()
        print("✅ Bar plot saved.")

    def save_single_dependence_plot(self, feature_name: str):
        """Save a dependence plot for a specific feature."""
        column_list = list(self.X_test.columns)
        feature_index = self.X_test.columns.get_loc(feature_name)

        X_test_values = self.X_test.values  # Convert to numpy

        shap.dependence_plot(
            ind=feature_index,
            shap_values=self.shap_values,
            features=X_test_values,
            feature_names=column_list,
            show=False
        )
        plt.title(f"Dependence Plot for {feature_name}")
        plt.savefig(os.path.join(self.output_dir, f"dependence_{feature_name}.png"), bbox_inches="tight")
        plt.close()
        print(f"✅ Dependence plot saved for feature '{feature_name}'.")
        
    def save_dependence_plots(self, top_n_features=5):
        """Save dependence plots for top N important features."""
        abs_shap_means = np.abs(self.shap_values).mean(axis=0)
        
        top_feature_indices = np.argsort(abs_shap_means)[::-1][:top_n_features]
        top_feature_indices = np.array(top_feature_indices).flatten().astype(int).tolist()  # 🔥 THE CORRECT FIX

        column_list = list(self.X_test.columns)
        top_features = [column_list[i] for i in top_feature_indices]

        X_test_values = self.X_test.values  # DataFrame to NumPy

        for feature in top_features:
            feature_index = self.X_test.columns.get_loc(feature)

            plt.figure()
            shap.dependence_plot(
                ind=feature_index,
                shap_values=self.shap_values,
                features=X_test_values,
                feature_names=column_list,
                show=False
            )
            plt.title(f"Dependence Plot for {feature}")
            plt.savefig(os.path.join(self.output_dir, f"dependence_{feature}.png"), bbox_inches="tight")
            plt.close()
            print(f"✅ Dependence plot saved for feature '{feature}'.")


    def full_generate_all_plots(self, top_n_features=5):
        """One-shot: Calculate SHAP values and save all plots."""
        self.calculate_shap_values()
        self.save_summary_plot()
        self.save_bar_plot()
        self.save_summary_plot_top_n_features(top_n=5)
        self.save_dependence_plots(top_n_features=top_n_features)
        print("✅ All SHAP plots generated and saved successfully.")

    def get_shap_values(self):
        """Return SHAP values and explainer object."""
        return self.shap_values, self.explainer
