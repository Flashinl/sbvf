"""
Comprehensive Evaluation Metrics for Multi-Horizon Predictions

Includes:
- Per-class precision, recall, F1-score for each horizon
- Calibration analysis (reliability diagrams)
- MAE/RMSE for return predictions
- Confusion matrices
- ROC curves and AUC scores
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_curve,
    auc,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MultiHorizonEvaluator:
    """Evaluator for multi-horizon predictions"""

    def __init__(self, horizons: List[str] = ['1week', '1month', '3month']):
        self.horizons = horizons
        self.class_names = ['SELL', 'HOLD', 'BUY']
        self.metrics = {}

    def compute_classification_metrics(
        self,
        predictions_dict: Dict[str, Dict[str, List]]
    ) -> Dict[str, Dict]:
        """
        Compute per-class metrics for each horizon

        Args:
            predictions_dict: Dict mapping horizon -> {
                'probs': list of probability arrays,
                'preds': list of predicted classes,
                'labels': list of true labels
            }

        Returns:
            Dict mapping horizon -> metrics dict
        """
        metrics = {}

        for horizon in self.horizons:
            y_true = np.array(predictions_dict[horizon]['labels'])
            y_pred = np.array(predictions_dict[horizon]['preds'])

            # Per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=[0, 1, 2], zero_division=0
            )

            # Overall accuracy
            accuracy = (y_true == y_pred).mean()

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

            metrics[horizon] = {
                'accuracy': float(accuracy),
                'precision': {self.class_names[i]: float(precision[i]) for i in range(3)},
                'recall': {self.class_names[i]: float(recall[i]) for i in range(3)},
                'f1': {self.class_names[i]: float(f1[i]) for i in range(3)},
                'support': {self.class_names[i]: int(support[i]) for i in range(3)},
                'confusion_matrix': cm.tolist()
            }

            # Macro averages
            metrics[horizon]['macro_precision'] = float(np.mean(precision))
            metrics[horizon]['macro_recall'] = float(np.mean(recall))
            metrics[horizon]['macro_f1'] = float(np.mean(f1))

        return metrics

    def compute_regression_metrics(
        self,
        predictions_dict: Dict[str, Dict[str, List]]
    ) -> Dict[str, Dict]:
        """
        Compute regression metrics for each horizon

        Args:
            predictions_dict: Dict mapping horizon -> {
                'regression_preds': list of predicted returns,
                'regression_labels': list of true returns
            }

        Returns:
            Dict mapping horizon -> metrics dict
        """
        metrics = {}

        for horizon in self.horizons:
            y_true = np.array(predictions_dict[horizon]['regression_labels'])
            y_pred = np.array(predictions_dict[horizon]['regression_preds'])

            # Remove any NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            if len(y_true) == 0:
                continue

            # MAE and RMSE
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            # Directional accuracy (did we predict the right direction?)
            direction_true = np.sign(y_true)
            direction_pred = np.sign(y_pred)
            directional_accuracy = (direction_true == direction_pred).mean()

            # R-squared
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            metrics[horizon] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'directional_accuracy': float(directional_accuracy),
                'r2': float(r2),
                'mean_error': float(np.mean(y_pred - y_true)),
                'std_error': float(np.std(y_pred - y_true))
            }

        return metrics

    def compute_calibration(
        self,
        predictions_dict: Dict[str, Dict[str, List]],
        n_bins: int = 10
    ) -> Dict[str, Dict]:
        """
        Compute calibration metrics (reliability diagrams)

        Calibration measures: Are 70% confidence predictions correct 70% of the time?

        Args:
            predictions_dict: Dict mapping horizon -> {
                'probs': list of probability arrays (shape: [N, 3]),
                'labels': list of true labels
            }
            n_bins: Number of bins for calibration curve

        Returns:
            Dict mapping horizon -> calibration metrics
        """
        calibration_metrics = {}

        for horizon in self.horizons:
            probs = np.array(predictions_dict[horizon]['probs'])  # Shape: (N, 3)
            labels = np.array(predictions_dict[horizon]['labels'])  # Shape: (N,)

            # Get max probability and predicted class for each sample
            max_probs = probs.max(axis=1)
            pred_classes = probs.argmax(axis=1)

            # Check if prediction was correct
            correct = (pred_classes == labels).astype(int)

            # Compute ECE (Expected Calibration Error)
            ece = 0.0
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_confidences = []
            bin_accuracies = []
            bin_counts = []

            for i in range(n_bins):
                bin_mask = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i + 1])

                if i == n_bins - 1:  # Include upper bound in last bin
                    bin_mask = (max_probs >= bin_edges[i]) & (max_probs <= bin_edges[i + 1])

                if bin_mask.sum() > 0:
                    bin_confidence = max_probs[bin_mask].mean()
                    bin_accuracy = correct[bin_mask].mean()
                    bin_count = bin_mask.sum()

                    bin_confidences.append(float(bin_confidence))
                    bin_accuracies.append(float(bin_accuracy))
                    bin_counts.append(int(bin_count))

                    # ECE contribution
                    ece += (bin_count / len(labels)) * abs(bin_accuracy - bin_confidence)
                else:
                    bin_confidences.append(None)
                    bin_accuracies.append(None)
                    bin_counts.append(0)

            calibration_metrics[horizon] = {
                'ece': float(ece),  # Expected Calibration Error
                'bin_confidences': bin_confidences,
                'bin_accuracies': bin_accuracies,
                'bin_counts': bin_counts,
                'bin_edges': bin_edges.tolist()
            }

        return calibration_metrics

    def compute_all_metrics(
        self,
        predictions_dict: Dict[str, Dict[str, List]]
    ) -> Dict:
        """Compute all metrics"""

        print("Computing classification metrics...")
        classification_metrics = self.compute_classification_metrics(predictions_dict)

        print("Computing regression metrics...")
        regression_metrics = self.compute_regression_metrics(predictions_dict)

        print("Computing calibration metrics...")
        calibration_metrics = self.compute_calibration(predictions_dict)

        return {
            'classification': classification_metrics,
            'regression': regression_metrics,
            'calibration': calibration_metrics
        }

    def print_metrics_report(self, metrics: Dict):
        """Print formatted metrics report"""

        print("\n" + "="*80)
        print("EVALUATION METRICS REPORT")
        print("="*80)

        for horizon in self.horizons:
            print(f"\n{'─'*80}")
            print(f"Horizon: {horizon.upper()}")
            print(f"{'─'*80}")

            # Classification metrics
            if horizon in metrics['classification']:
                cls_metrics = metrics['classification'][horizon]
                print(f"\nClassification Metrics:")
                print(f"  Overall Accuracy: {cls_metrics['accuracy']:.3f}")
                print(f"  Macro F1-Score:   {cls_metrics['macro_f1']:.3f}")

                print(f"\n  Per-Class Metrics:")
                print(f"  {'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
                print(f"  {'-'*60}")
                for class_name in self.class_names:
                    print(f"  {class_name:<8} "
                          f"{cls_metrics['precision'][class_name]:>10.3f}  "
                          f"{cls_metrics['recall'][class_name]:>10.3f}  "
                          f"{cls_metrics['f1'][class_name]:>10.3f}  "
                          f"{cls_metrics['support'][class_name]:>8}")

            # Regression metrics
            if horizon in metrics['regression']:
                reg_metrics = metrics['regression'][horizon]
                print(f"\n  Regression Metrics:")
                print(f"  MAE:                  {reg_metrics['mae']:.3f}%")
                print(f"  RMSE:                 {reg_metrics['rmse']:.3f}%")
                print(f"  Directional Accuracy: {reg_metrics['directional_accuracy']:.3f}")
                print(f"  R²:                   {reg_metrics['r2']:.3f}")

            # Calibration metrics
            if horizon in metrics['calibration']:
                cal_metrics = metrics['calibration'][horizon]
                print(f"\n  Calibration:")
                print(f"  ECE (Expected Calibration Error): {cal_metrics['ece']:.4f}")
                print(f"  (Lower is better, 0 = perfectly calibrated)")

        print("\n" + "="*80)

    def plot_confusion_matrices(
        self,
        metrics: Dict,
        save_path: Optional[str] = None
    ):
        """Plot confusion matrices for each horizon"""

        fig, axes = plt.subplots(1, len(self.horizons), figsize=(15, 4))
        if len(self.horizons) == 1:
            axes = [axes]

        for idx, horizon in enumerate(self.horizons):
            cm = np.array(metrics['classification'][horizon]['confusion_matrix'])

            # Normalize by row (true labels)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(
                cm_normalized,
                annot=cm,  # Show raw counts
                fmt='d',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=axes[idx],
                cbar_kws={'label': 'Normalized Frequency'}
            )

            axes[idx].set_title(f'{horizon} Confusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrices saved to: {save_path}")

        plt.show()

    def plot_calibration_curves(
        self,
        metrics: Dict,
        save_path: Optional[str] = None
    ):
        """Plot calibration curves (reliability diagrams)"""

        fig, axes = plt.subplots(1, len(self.horizons), figsize=(15, 4))
        if len(self.horizons) == 1:
            axes = [axes]

        for idx, horizon in enumerate(self.horizons):
            cal_metrics = metrics['calibration'][horizon]

            bin_confidences = [c for c in cal_metrics['bin_confidences'] if c is not None]
            bin_accuracies = [a for a in cal_metrics['bin_accuracies'] if a is not None]

            # Plot calibration curve
            axes[idx].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5)
            axes[idx].plot(bin_confidences, bin_accuracies, 'o-', label='Model', markersize=8)

            # Add ECE annotation
            ece = cal_metrics['ece']
            axes[idx].text(
                0.05, 0.95,
                f'ECE: {ece:.4f}',
                transform=axes[idx].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

            axes[idx].set_xlabel('Confidence (Predicted Probability)')
            axes[idx].set_ylabel('Accuracy (Actual Correctness)')
            axes[idx].set_title(f'{horizon} Calibration Curve')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
            axes[idx].set_xlim([0, 1])
            axes[idx].set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Calibration curves saved to: {save_path}")

        plt.show()

    def plot_regression_scatter(
        self,
        predictions_dict: Dict[str, Dict[str, List]],
        save_path: Optional[str] = None
    ):
        """Plot predicted vs actual returns"""

        fig, axes = plt.subplots(1, len(self.horizons), figsize=(15, 4))
        if len(self.horizons) == 1:
            axes = [axes]

        for idx, horizon in enumerate(self.horizons):
            y_true = np.array(predictions_dict[horizon]['regression_labels'])
            y_pred = np.array(predictions_dict[horizon]['regression_preds'])

            # Remove NaN
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            # Scatter plot with density
            axes[idx].hexbin(y_true, y_pred, gridsize=30, cmap='Blues', mincnt=1)
            axes[idx].plot([-50, 50], [-50, 50], 'r--', alpha=0.5, label='Perfect Prediction')

            # MAE annotation
            mae = np.abs(y_true - y_pred).mean()
            axes[idx].text(
                0.05, 0.95,
                f'MAE: {mae:.2f}%',
                transform=axes[idx].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

            axes[idx].set_xlabel('True Return (%)')
            axes[idx].set_ylabel('Predicted Return (%)')
            axes[idx].set_title(f'{horizon} Return Predictions')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Regression scatter plots saved to: {save_path}")

        plt.show()


def evaluate_model_predictions(
    predictions_dict: Dict[str, Dict[str, List]],
    output_dir: Optional[str] = None,
    show_plots: bool = True
) -> Dict:
    """
    Complete evaluation pipeline

    Args:
        predictions_dict: Predictions from model
        output_dir: Directory to save plots
        show_plots: Whether to display plots

    Returns:
        Complete metrics dictionary
    """
    evaluator = MultiHorizonEvaluator()

    # Compute all metrics
    metrics = evaluator.compute_all_metrics(predictions_dict)

    # Print report
    evaluator.print_metrics_report(metrics)

    # Generate plots
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrices
        evaluator.plot_confusion_matrices(
            metrics,
            save_path=output_dir / 'confusion_matrices.png'
        )

        # Calibration curves
        evaluator.plot_calibration_curves(
            metrics,
            save_path=output_dir / 'calibration_curves.png'
        )

        # Regression scatter
        evaluator.plot_regression_scatter(
            predictions_dict,
            save_path=output_dir / 'regression_scatter.png'
        )

        # Save metrics to JSON
        import json
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\nMetrics and plots saved to: {output_dir}")

    elif show_plots:
        evaluator.plot_confusion_matrices(metrics)
        evaluator.plot_calibration_curves(metrics)
        evaluator.plot_regression_scatter(predictions_dict)

    return metrics


if __name__ == '__main__':
    # Example usage with dummy data
    print("Testing Evaluation Module")
    print("="*60)

    # Create dummy predictions
    np.random.seed(42)
    n_samples = 1000
    horizons = ['1week', '1month', '3month']

    predictions_dict = {}
    for horizon in horizons:
        # Simulate predictions
        true_labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
        predicted_probs = np.random.dirichlet([2, 3, 2], size=n_samples)  # Realistic probabilities
        predicted_classes = predicted_probs.argmax(axis=1)

        # Simulate some correlation with truth (not perfect)
        accuracy = 0.6
        for i in range(n_samples):
            if np.random.rand() > accuracy:
                predicted_classes[i] = true_labels[i]

        # Regression targets
        true_returns = np.random.normal(0, 5, size=n_samples)
        predicted_returns = true_returns + np.random.normal(0, 3, size=n_samples)

        predictions_dict[horizon] = {
            'probs': predicted_probs.tolist(),
            'preds': predicted_classes.tolist(),
            'labels': true_labels.tolist(),
            'regression_preds': predicted_returns.tolist(),
            'regression_labels': true_returns.tolist()
        }

    # Evaluate
    metrics = evaluate_model_predictions(
        predictions_dict,
        output_dir='evaluation_test',
        show_plots=False
    )

    print("\nTest complete!")
