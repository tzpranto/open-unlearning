#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for SIBL (Resumable with Evaluation Metrics)
==========================================================================

This script automates hyperparameter tuning for the SIBL unlearning method.
It performs a grid search over specified hyperparameters, trains models,
evaluates them, and tracks key evaluation metrics (Model Utility, Forget Quality, Privacy Leak).

Key Features:
- Resumable: Automatically resumes from where it left off if interrupted
- Integrated evaluation: Runs evaluation after training and extracts key metrics
- Tracks three eval metrics: Model Utility, Forget Quality, and Privacy Leak
- Immediate saving: Results are saved after each run to prevent data loss
- Error handling: Failed runs are recorded and the search continues

Output Files:
- results/sibl_tuning/tuning_results.csv: All runs with hyperparameters and evaluation metrics
- results/sibl_tuning/tuning_state.json: Resumable state
- saves/unlearn/sibl_tune_run_XXXX/: Model checkpoints
- saves/unlearn/sibl_tune_run_XXXX/evals/TOFU_*.json: Evaluation results

Usage:
    # Start or resume tuning
    python tune_sibl_hyperparameters.py --output_dir results/sibl_tuning

    # Run limited subset
    python tune_sibl_hyperparameters.py --max_runs 50 --sample_random
"""

import argparse
import itertools
import sys
import subprocess
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Manages hyperparameter tuning for SIBL."""

    def __init__(self, output_dir: str):
        """
        Initialize the tuner.

        Args:
            output_dir: Directory to store results (e.g., results/sibl_tuning)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = []
        self.results_csv = self.output_dir / "tuning_results.csv"
        self.state_file = self.output_dir / "tuning_state.json"
        self.completed_runs = set()

        # Load existing results and state if resuming
        self._load_existing_results()

    def _load_existing_results(self):
        """Load existing results and state for resuming interrupted runs."""
        has_existing = False

        if self.results_csv.exists():
            try:
                df = pd.read_csv(self.results_csv)
                self.results = df.to_dict('records')
                self.completed_runs = set(df['run_id'].astype(int).tolist())
                has_existing = True
                logger.info("="*80)
                logger.info("RESUMING FROM PREVIOUS RUN")
                logger.info("="*80)
                logger.info(f"Loaded {len(self.results)} existing results")
                logger.info(f"Completed runs: {len(self.completed_runs)}")
                successful = sum(1 for r in self.results if r.get('success', False))
                failed = len(self.results) - successful
                logger.info(f"  Successful: {successful}")
                logger.info(f"  Failed: {failed}")
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")

        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.completed_runs.update(state.get('completed_runs', []))
                if not has_existing:
                    logger.info(f"Loaded state file with {len(self.completed_runs)} completed runs")
                last_updated = state.get('last_updated', 'unknown')
                logger.info(f"Last updated: {last_updated}")
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}")

        if has_existing:
            logger.info("Runs already completed will be skipped")
            logger.info("="*80 + "\n")

    def _save_state(self):
        """Save current state for resuming later."""
        state = {
            'completed_runs': sorted(list(self.completed_runs)),
            'last_updated': datetime.now().isoformat(),
            'total_runs': len(self.results)
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save state file: {e}")

    def define_parameter_grid(self) -> Dict[str, List[Any]]:
        """
        Define the hyperparameter search grid.

        Returns:
            Dictionary mapping parameter names to lists of values to try
        """
        grid = {
            # Sparsity parameters
            'use_sparsity': [False, True],
            'sparsity': [0.1, 0.3],
            'sparsity_method': [
                'layerwise_magnitude',
                'preserve_attn_embed',
            ],

            # SIBL parameters
            'epsilon': [0.1, 0.075],  # Retain loss budget
            'T': [10],  # Outer iterations (fixed)
            'K': [10],  # Inner iterations (fixed)
            'eta_in': [1e-4],  # Inner LR
            'eta_theta_multiplier': [2.0, 1.5, 1.75],  # Multiplier for eta_theta
            'rho': [1.0]  # AL penalty
            'gamma': [2e-4],  # L1 regularization
            'use_implicit': [False],  # Keep false for now
            'cg_iters': [10],  # Fixed
            'cg_tol': [1e-3],  # Fixed
        }

        return grid

    def run_experiment(self, run_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment with the given parameters.

        Args:
            run_id: Unique identifier for this run
            params: Parameter values for this run

        Returns:
            Dictionary containing run results and metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting Run {run_id}")
        logger.info(f"Parameters: {params}")
        logger.info(f"{'='*80}\n")

        # Prepare task name
        task_name = f"sibl_tune_run_{run_id:04d}"

        # Build hyperparameter overrides for training
        eta_theta = params['eta_in'] * params['eta_theta_multiplier']

        # Training command - following tofu_baselines.sh pattern
        train_cmd = [
            sys.executable,
            "src/train.py",
            "--config-name=unlearn.yaml",
            f"experiment=unlearn/tofu/sibl.yaml",
            f"trainer=SIBL",
            f"task_name={task_name}",
            f"model=Llama-3.2-1B-Instruct",
            f"forget_split=forget01",
            f"retain_split=retain99",
            f"trainer.method_args.use_sparsity={str(params['use_sparsity']).lower()}",
            f"trainer.method_args.sparsity={params['sparsity']}",
            f"trainer.method_args.sparsity_method={params['sparsity_method']}",
            f"trainer.method_args.epsilon={params['epsilon']}",
            f"trainer.method_args.T={params['T']}",
            f"trainer.method_args.K={params['K']}",
            f"trainer.method_args.eta_theta={eta_theta}",
            f"trainer.method_args.eta_in={params['eta_in']}",
            f"trainer.method_args.rho={params['rho']}",
            f"trainer.method_args.gamma={params['gamma']}",
            f"trainer.method_args.use_implicit={str(params['use_implicit']).lower()}",
            f"trainer.method_args.cg_iters={params['cg_iters']}",
            f"trainer.method_args.cg_tol={params['cg_tol']}",
        ]

        # Run training
        start_time = datetime.now()
        success = False
        metrics = {}

        try:
            logger.info(f"Training step for run {run_id}...")
            result = subprocess.run(
                train_cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"Run {run_id} training failed with error:")
                logger.error(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
            else:
                logger.info(f"Run {run_id} training completed successfully")

                # Run evaluation - following tofu_baselines.sh pattern
                logger.info(f"Evaluation step for run {run_id}...")
                retain_split = 'retain99'
                forget_split = 'forget01'
                holdout_split = 'holdout01'
                model = 'Llama-3.2-1B-Instruct'
                eval_cmd = [
                    sys.executable,
                    "src/eval.py",
                    "experiment=eval/tofu/default.yaml",
                    f"forget_split=forget01",
                    f"holdout_split=holdout01",
                    f"model=Llama-3.2-1B-Instruct",
                    f"task_name={task_name}",
                    f"model.model_args.pretrained_model_name_or_path=saves/unlearn/{task_name}",
                    f"paths.output_dir=saves/unlearn/{task_name}/evals",
                    f"retain_logs_path=saves/eval/tofu_{model}_{retain_split}/TOFU_EVAL.json"
                ]

                eval_result = subprocess.run(
                    eval_cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout for eval
                )

                if eval_result.returncode != 0:
                    logger.warning(f"Run {run_id} evaluation failed, but training succeeded")
                    logger.warning(eval_result.stderr[-1000:] if len(eval_result.stderr) > 1000 else eval_result.stderr)
                    success = True
                else:
                    logger.info(f"Run {run_id} evaluation completed successfully")
                    success = True
                    # Extract metrics from evaluation output
                    metrics = self.extract_metrics(task_name)

        except subprocess.TimeoutExpired:
            logger.error(f"Run {run_id} timed out")
            success = False
        except Exception as e:
            logger.error(f"Run {run_id} failed with exception: {e}")
            success = False

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Compile results
        result_dict = {
            'run_id': run_id,
            'success': success,
            'duration_seconds': duration,
            'timestamp': start_time.isoformat(),
            # Parameters
            'use_sparsity': params['use_sparsity'],
            'sparsity': params['sparsity'],
            'sparsity_method': params['sparsity_method'],
            'epsilon': params['epsilon'],
            'T': params['T'],
            'K': params['K'],
            'eta_theta': eta_theta,
            'eta_in': params['eta_in'],
            'eta_theta_multiplier': params['eta_theta_multiplier'],
            'rho': params['rho'],
            'gamma': params['gamma'],
            'use_implicit': params['use_implicit'],
            'cg_iters': params['cg_iters'],
            'cg_tol': params['cg_tol'],
            # Metrics
            'model_utility': metrics.get('model_utility'),
            'forget_quality': metrics.get('forget_quality'),
            'privacy_leak': metrics.get('privacy_leak'),
        }

        return result_dict

    def extract_metrics(self, task_name: str) -> Dict[str, Any]:
        """
        Extract evaluation metrics from evaluation output JSON files.

        Args:
            task_name: Task name to locate eval results

        Returns:
            Dictionary of metrics with keys: model_utility, forget_quality, privacy_leak
        """
        metrics = {}

        # Look for evaluation results in the standard output directory
        eval_summary_path = Path(f"./saves/unlearn/{task_name}/evals/TOFU_SUMMARY.json")
        eval_detailed_path = Path(f"./saves/unlearn/{task_name}/evals/TOFU_EVAL.json")

        # Try to load summary first (has aggregated values)
        if eval_summary_path.exists():
            try:
                with open(eval_summary_path, 'r') as f:
                    summary_metrics = json.load(f)

                # Extract the three key metrics
                if 'model_utility' in summary_metrics:
                    metrics['model_utility'] = summary_metrics['model_utility']
                if 'forget_quality' in summary_metrics:
                    metrics['forget_quality'] = summary_metrics['forget_quality']
                if 'privleak' in summary_metrics:
                    metrics['privacy_leak'] = summary_metrics['privleak']

                logger.info(f"Loaded metrics from {eval_summary_path}")
                logger.info(f"  Model Utility: {metrics.get('model_utility')}")
                logger.info(f"  Forget Quality: {metrics.get('forget_quality')}")
                logger.info(f"  Privacy Leak: {metrics.get('privacy_leak')}")
                return metrics

            except Exception as e:
                logger.warning(f"Failed to load summary metrics: {e}")

        # Fall back to detailed eval file if summary doesn't exist
        if eval_detailed_path.exists():
            try:
                with open(eval_detailed_path, 'r') as f:
                    detailed_metrics = json.load(f)

                # Extract aggregated values from detailed metrics
                if 'model_utility' in detailed_metrics:
                    metrics['model_utility'] = detailed_metrics['model_utility'].get('agg_value')
                if 'forget_quality' in detailed_metrics:
                    metrics['forget_quality'] = detailed_metrics['forget_quality'].get('agg_value')
                if 'privleak' in detailed_metrics:
                    metrics['privacy_leak'] = detailed_metrics['privleak'].get('agg_value')

                logger.info(f"Loaded metrics from {eval_detailed_path}")
                logger.info(f"  Model Utility: {metrics.get('model_utility')}")
                logger.info(f"  Forget Quality: {metrics.get('forget_quality')}")
                logger.info(f"  Privacy Leak: {metrics.get('privacy_leak')}")
                return metrics

            except Exception as e:
                logger.warning(f"Failed to load detailed metrics: {e}")

        if not metrics:
            logger.warning(f"No evaluation metrics found for {task_name}")

        return metrics

    def run_grid_search(self, max_runs: int = None, sample_random: bool = False):
        """
        Run grid search over hyperparameter space.

        Args:
            max_runs: Maximum number of runs (None for all combinations)
            sample_random: If True, sample random combinations instead of grid search
        """
        grid = self.define_parameter_grid()

        # Generate all parameter combinations
        param_names = list(grid.keys())
        param_values = [grid[name] for name in param_names]
        all_combinations = list(itertools.product(*param_values))

        logger.info(f"Total parameter combinations: {len(all_combinations)}")

        if max_runs and max_runs < len(all_combinations):
            if sample_random:
                all_combinations = random.sample(all_combinations, max_runs)
                logger.info(f"Randomly sampling {max_runs} combinations")
            else:
                all_combinations = all_combinations[:max_runs]
                logger.info(f"Running first {max_runs} combinations")

        # Count how many runs to skip
        skipped = 0
        completed = 0
        failed = 0

        # Run experiments
        for run_id, combo in enumerate(all_combinations, start=1):
            params = dict(zip(param_names, combo))

            # Check if this run has already been completed
            if run_id in self.completed_runs:
                skipped += 1
                logger.info(f"Skipping run {run_id} (already completed)")
                continue

            # Run experiment
            try:
                result = self.run_experiment(run_id, params)

                # Store result
                self.results.append(result)

                # Mark as completed
                self.completed_runs.add(run_id)

                # Save results and state immediately
                self.save_results()
                self._save_state()

                if result['success']:
                    completed += 1
                else:
                    failed += 1

                logger.info(f"Progress: {len(self.completed_runs)}/{len(all_combinations)} runs "
                          f"(completed: {completed}, failed: {failed}, skipped: {skipped})")

            except KeyboardInterrupt:
                logger.warning("\nKeyboard interrupt detected. Saving progress...")
                self.save_results()
                self._save_state()
                logger.info(f"Progress saved. You can resume later by running the same command.")
                raise

            except Exception as e:
                logger.error(f"Unexpected error in run {run_id}: {e}")
                failed_result = {
                    'run_id': run_id,
                    'success': False,
                    'duration_seconds': 0,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    **params
                }
                self.results.append(failed_result)
                self.completed_runs.add(run_id)
                failed += 1

                self.save_results()
                self._save_state()

                logger.info(f"Progress: {len(self.completed_runs)}/{len(all_combinations)} runs "
                          f"(completed: {completed}, failed: {failed}, skipped: {skipped})")

        logger.info("\n" + "="*80)
        logger.info("Grid search completed!")
        logger.info(f"Total runs: {len(all_combinations)}")
        logger.info(f"Completed successfully: {completed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped (already done): {skipped}")
        logger.info(f"Results saved to: {self.results_csv}")
        logger.info(f"State saved to: {self.state_file}")
        logger.info("="*80)

    def save_results(self):
        """Save results to CSV file immediately with error handling."""
        if not self.results:
            return

        try:
            df = pd.DataFrame(self.results)
            # Write to temporary file first to avoid corruption
            temp_csv = self.results_csv.with_suffix('.tmp')
            df.to_csv(temp_csv, index=False)
            # Atomic rename (on most systems)
            temp_csv.replace(self.results_csv)
            logger.debug(f"Results saved to {self.results_csv}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def analyze_results(self):
        """Analyze and summarize results."""
        if not self.results:
            logger.warning("No results to analyze")
            return

        df = pd.DataFrame(self.results)

        # Filter successful runs with metrics
        successful_runs = df[df['success'] == True].copy()
        runs_with_metrics = successful_runs.dropna(subset=['model_utility', 'forget_quality', 'privacy_leak'], how='all')

        logger.info("\n" + "="*80)
        logger.info("RESULTS SUMMARY")
        logger.info("="*80)
        logger.info(f"Total runs: {len(df)}")
        logger.info(f"Successful runs: {len(successful_runs)}")
        logger.info(f"Runs with evaluation metrics: {len(runs_with_metrics)}")
        logger.info(f"Failed runs: {len(df) - len(successful_runs)}")

        if len(runs_with_metrics) == 0:
            logger.warning("No successful runs with evaluation metrics to analyze")
            return

        # Find best configurations for each metric
        logger.info("\n" + "="*80)
        logger.info("BEST CONFIGURATIONS BY METRIC")
        logger.info("="*80)

        # Best Model Utility (higher is better)
        if 'model_utility' in runs_with_metrics.columns:
            valid_utility = runs_with_metrics.dropna(subset=['model_utility'])
            if len(valid_utility) > 0:
                best_utility = valid_utility.loc[valid_utility['model_utility'].idxmax()]
                logger.info("\nBest Model Utility:")
                logger.info(f"  Run ID: {int(best_utility['run_id'])}")
                logger.info(f"  Model Utility: {best_utility['model_utility']:.4f}")
                logger.info(f"  Forget Quality: {best_utility['forget_quality']:.4f}")
                logger.info(f"  Privacy Leak: {best_utility['privacy_leak']:.4f}")
                logger.info(f"  Hyperparameters:")
                logger.info(f"    Epsilon: {best_utility['epsilon']:.3f}")
                logger.info(f"    Rho: {best_utility['rho']:.2f}")
                logger.info(f"    Gamma: {best_utility['gamma']:.2e}")
                logger.info(f"    Sparsity: {best_utility['sparsity']:.2f}")

        # Best Forget Quality (higher p-value = better)
        if 'forget_quality' in runs_with_metrics.columns:
            valid_fq = runs_with_metrics.dropna(subset=['forget_quality'])
            if len(valid_fq) > 0:
                best_fq = valid_fq.loc[valid_fq['forget_quality'].idxmax()]
                logger.info("\nBest Forget Quality:")
                logger.info(f"  Run ID: {int(best_fq['run_id'])}")
                logger.info(f"  Forget Quality: {best_fq['forget_quality']:.4f}")
                logger.info(f"  Model Utility: {best_fq['model_utility']:.4f}")
                logger.info(f"  Privacy Leak: {best_fq['privacy_leak']:.4f}")
                logger.info(f"  Hyperparameters:")
                logger.info(f"    Epsilon: {best_fq['epsilon']:.3f}")
                logger.info(f"    Rho: {best_fq['rho']:.2f}")
                logger.info(f"    Gamma: {best_fq['gamma']:.2e}")
                logger.info(f"    Sparsity: {best_fq['sparsity']:.2f}")

        # Best Privacy (lower is better - lower privacy leak = better)
        if 'privacy_leak' in runs_with_metrics.columns:
            valid_priv = runs_with_metrics.dropna(subset=['privacy_leak'])
            if len(valid_priv) > 0:
                best_priv = valid_priv.loc[valid_priv['privacy_leak'].idxmin()]
                logger.info("\nBest Privacy (Lowest Leak):")
                logger.info(f"  Run ID: {int(best_priv['run_id'])}")
                logger.info(f"  Privacy Leak: {best_priv['privacy_leak']:.4f}")
                logger.info(f"  Model Utility: {best_priv['model_utility']:.4f}")
                logger.info(f"  Forget Quality: {best_priv['forget_quality']:.4f}")
                logger.info(f"  Hyperparameters:")
                logger.info(f"    Epsilon: {best_priv['epsilon']:.3f}")
                logger.info(f"    Rho: {best_priv['rho']:.2f}")
                logger.info(f"    Gamma: {best_priv['gamma']:.2e}")
                logger.info(f"    Sparsity: {best_priv['sparsity']:.2f}")

        logger.info("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for SIBL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a new tuning run
  python tune_sibl_hyperparameters.py --output_dir results/sibl_tuning

  # Resume interrupted run
  python tune_sibl_hyperparameters.py --output_dir results/sibl_tuning

  # Run only 20 random combinations
  python tune_sibl_hyperparameters.py --output_dir results/sibl_tuning --max_runs 20 --sample_random
        """
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sibl_tuning",
        help="Directory to store results"
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Maximum number of runs (None for all combinations)"
    )
    parser.add_argument(
        "--sample_random",
        action="store_true",
        help="Sample random combinations instead of grid search"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create tuner
    tuner = HyperparameterTuner(output_dir=args.output_dir)

    # Run grid search
    tuner.run_grid_search(
        max_runs=args.max_runs,
        sample_random=args.sample_random
    )

    # Analyze results
    tuner.analyze_results()


if __name__ == "__main__":
    main()