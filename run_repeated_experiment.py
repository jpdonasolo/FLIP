#!/usr/bin/env python3
"""
Run resenet18_cifar100 experiment multiple times and average metrics across budgets.
Saves intermediate results after each run and final averaged metrics.
"""

import os
import sys
import subprocess
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# Add modules to path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'modules')
))

from modules.base_utils import util

EXPERIMENT_NAME = "resenet18_cifar100"
NUM_RUNS = 10
BUDGETS = [150, 300, 500, 1000, 1500]

# Output directories
RESULTS_DIR = f"experiments/{EXPERIMENT_NAME}/repeated_runs"
AVG_RESULTS_DIR = f"experiments/{EXPERIMENT_NAME}/averaged_results"


def run_module(experiment_name, module_name):
    """Run a single module from the experiment."""
    try:
        result = subprocess.run(
            [sys.executable, "run_experiment.py", experiment_name],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            print(f"Error running {module_name}:")
            print(result.stderr)
            return False
        return True
    except Exception as e:
        print(f"Exception running {module_name}: {e}")
        return False


def run_single_module(experiment_name, module_name):
    """Run a specific module by importing and calling it directly."""
    try:
        module_file = module_name
        # Check for internal module name mapping
        schema_path = util.generate_full_path(f"schemas/{module_name}.toml")
        if os.path.exists(schema_path):
            import toml
            from collections import OrderedDict
            schema = toml.load(schema_path, _dict=OrderedDict)
            if 'INTERNAL' in schema:
                module_file = schema['INTERNAL']['module_name']
        
        module = __import__(f"{module_file}", fromlist=["run_module"])
        module.run_module.run(experiment_name, module_name)
        return True
    except Exception as e:
        print(f"Exception running {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_metrics(run_dir, budget):
    """Load metrics for a specific budget from a run directory."""
    budget_file = f"experiments/{EXPERIMENT_NAME}/{budget}.npy"
    downstream_dir = f"{run_dir}/budget_{budget}/"
    
    metrics = {}
    
    # Load downstream metrics if they exist
    paccs_file = f"{downstream_dir}/paccs.npy"
    caccs_file = f"{downstream_dir}/caccs.npy"
    
    if os.path.exists(paccs_file):
        poison_metrics = np.load(paccs_file, allow_pickle=True)
        # poison_metrics is a list of (acc, loss) tuples per epoch
        if len(poison_metrics) > 0:
            # Get final epoch metrics
            final_pacc, final_ploss = poison_metrics[-1]
            metrics['final_poison_acc'] = float(final_pacc)
            metrics['final_poison_loss'] = float(final_ploss)
            # Get all epoch accuracies
            metrics['poison_accs'] = [float(acc) for acc, _ in poison_metrics]
    
    if os.path.exists(caccs_file):
        clean_metrics = np.load(caccs_file, allow_pickle=True)
        # clean_metrics is a list of (acc, loss) tuples per epoch
        if len(clean_metrics) > 0:
            # Get final epoch metrics
            final_cacc, final_closs = clean_metrics[-1]
            metrics['final_clean_acc'] = float(final_cacc)
            metrics['final_clean_loss'] = float(final_closs)
            # Get all epoch accuracies
            metrics['clean_accs'] = [float(acc) for acc, _ in clean_metrics]
    
    return metrics


def run_train_user_for_budget(budget, output_dir, run_id):
    """Run train_user module for a specific budget."""
    import toml
    from collections import OrderedDict
    
    config_path = util.generate_full_path(f"experiments/{EXPERIMENT_NAME}/config.toml")
    config = toml.load(config_path, _dict=OrderedDict)
    
    # Modify train_user section for this budget
    config['train_user']['input_labels'] = f"experiments/{EXPERIMENT_NAME}/{budget}.npy"
    config['train_user']['output_dir'] = output_dir
    
    # Temporarily modify the config file
    original_config = toml.load(config_path, _dict=OrderedDict)
    try:
        with open(config_path, 'w') as f:
            toml.dump(config, f)
        
        success = run_single_module(EXPERIMENT_NAME, "train_user")
        return success
    finally:
        # Restore original config
        with open(config_path, 'w') as f:
            toml.dump(original_config, f)


def run_experiment_with_budgets(run_id):
    """Run the full experiment for a single run, including train_user for all budgets."""
    print(f"\n{'='*60}")
    print(f"Starting Run {run_id + 1}/{NUM_RUNS}")
    print(f"{'='*60}\n")
    
    run_dir = f"{RESULTS_DIR}/run_{run_id:02d}"
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Train expert (only on first run, or if not exists)
    expert_dir = "out/checkpoints/r18_cifar100_1xs/0/"
    if run_id == 0 or not os.path.exists(expert_dir) or len(os.listdir(expert_dir)) == 0:
        print(f"Run {run_id + 1}: Training expert...")
        if not run_single_module(EXPERIMENT_NAME, "train_expert"):
            print(f"Failed to train expert in run {run_id + 1}")
            return None
    else:
        print(f"Run {run_id + 1}: Skipping expert training (already exists)")
    
    # Step 2: Generate labels (only on first run, or if not exists)
    labels_file = f"experiments/{EXPERIMENT_NAME}/labels.npy"
    if run_id == 0 or not os.path.exists(labels_file):
        print(f"Run {run_id + 1}: Generating labels...")
        if not run_single_module(EXPERIMENT_NAME, "generate_labels"):
            print(f"Failed to generate labels in run {run_id + 1}")
            return None
    else:
        print(f"Run {run_id + 1}: Skipping label generation (already exists)")
    
    # Step 3: Select flips (only on first run, or if not exists)
    budgets_exist = all(os.path.exists(f"experiments/{EXPERIMENT_NAME}/{budget}.npy") for budget in BUDGETS)
    if run_id == 0 or not budgets_exist:
        print(f"Run {run_id + 1}: Selecting flips...")
        if not run_single_module(EXPERIMENT_NAME, "select_flips"):
            print(f"Failed to select flips in run {run_id + 1}")
            return None
    else:
        print(f"Run {run_id + 1}: Skipping flip selection (already exists)")
    
    # Step 4: Train user models for each budget
    run_metrics = {}
    for budget in BUDGETS:
        print(f"Run {run_id + 1}: Training user model for budget {budget}...")
        
        budget_downstream_dir = f"{run_dir}/budget_{budget}/"
        Path(budget_downstream_dir).mkdir(parents=True, exist_ok=True)
        
        if run_train_user_for_budget(budget, budget_downstream_dir, run_id):
            # Load metrics
            metrics = load_metrics(run_dir, budget)
            run_metrics[budget] = metrics
            clean_acc = metrics.get('final_clean_acc', None)
            poison_acc = metrics.get('final_poison_acc', None)
            clean_str = f"{clean_acc:.4f}" if clean_acc is not None else "N/A"
            poison_str = f"{poison_acc:.4f}" if poison_acc is not None else "N/A"
            print(f"  Budget {budget}: Clean Acc = {clean_str}, Poison Acc = {poison_str}")
        else:
            print(f"  Failed to train user model for budget {budget} in run {run_id + 1}")
            run_metrics[budget] = None
    
    # Save run metrics
    metrics_file = f"{run_dir}/metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(run_metrics, f, indent=2)
    
    print(f"\nRun {run_id + 1} completed. Metrics saved to {metrics_file}")
    return run_metrics


def average_metrics(all_run_metrics):
    """Average metrics across all runs for each budget."""
    averaged = {}
    
    for budget in BUDGETS:
        budget_metrics = [run_metrics.get(budget) for run_metrics in all_run_metrics if run_metrics and budget in run_metrics and run_metrics[budget] is not None]
        
        if len(budget_metrics) == 0:
            continue
        
        avg_metrics = {}
        
        # Average final metrics
        final_clean_accs = [m.get('final_clean_acc') for m in budget_metrics if 'final_clean_acc' in m]
        final_poison_accs = [m.get('final_poison_acc') for m in budget_metrics if 'final_poison_acc' in m]
        final_clean_losses = [m.get('final_clean_loss') for m in budget_metrics if 'final_clean_loss' in m]
        final_poison_losses = [m.get('final_poison_loss') for m in budget_metrics if 'final_poison_loss' in m]
        
        if final_clean_accs:
            avg_metrics['final_clean_acc'] = {
                'mean': float(np.mean(final_clean_accs)),
                'std': float(np.std(final_clean_accs)),
                'min': float(np.min(final_clean_accs)),
                'max': float(np.max(final_clean_accs))
            }
        
        if final_poison_accs:
            avg_metrics['final_poison_acc'] = {
                'mean': float(np.mean(final_poison_accs)),
                'std': float(np.std(final_poison_accs)),
                'min': float(np.min(final_poison_accs)),
                'max': float(np.max(final_poison_accs))
            }
        
        if final_clean_losses:
            avg_metrics['final_clean_loss'] = {
                'mean': float(np.mean(final_clean_losses)),
                'std': float(np.std(final_clean_losses))
            }
        
        if final_poison_losses:
            avg_metrics['final_poison_loss'] = {
                'mean': float(np.mean(final_poison_losses)),
                'std': float(np.std(final_poison_losses))
            }
        
        # Average per-epoch accuracies (if available)
        # Find max epochs
        max_epochs = 0
        for m in budget_metrics:
            if 'clean_accs' in m:
                max_epochs = max(max_epochs, len(m['clean_accs']))
        
        if max_epochs > 0:
            clean_accs_per_epoch = []
            poison_accs_per_epoch = []
            
            for epoch in range(max_epochs):
                epoch_clean_accs = []
                epoch_poison_accs = []
                
                for m in budget_metrics:
                    if 'clean_accs' in m and epoch < len(m['clean_accs']):
                        epoch_clean_accs.append(m['clean_accs'][epoch])
                    if 'poison_accs' in m and epoch < len(m['poison_accs']):
                        epoch_poison_accs.append(m['poison_accs'][epoch])
                
                if epoch_clean_accs:
                    clean_accs_per_epoch.append({
                        'epoch': epoch + 1,
                        'mean': float(np.mean(epoch_clean_accs)),
                        'std': float(np.std(epoch_clean_accs))
                    })
                
                if epoch_poison_accs:
                    poison_accs_per_epoch.append({
                        'epoch': epoch + 1,
                        'mean': float(np.mean(epoch_poison_accs)),
                        'std': float(np.std(epoch_poison_accs))
                    })
            
            if clean_accs_per_epoch:
                avg_metrics['clean_accs_per_epoch'] = clean_accs_per_epoch
            if poison_accs_per_epoch:
                avg_metrics['poison_accs_per_epoch'] = poison_accs_per_epoch
        
        avg_metrics['num_runs'] = len(budget_metrics)
        averaged[budget] = avg_metrics
    
    return averaged


def main():
    """Main function to run experiments and average results."""
    print(f"Starting repeated experiment: {EXPERIMENT_NAME}")
    print(f"Number of runs: {NUM_RUNS}")
    print(f"Budgets: {BUDGETS}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Averaged results directory: {AVG_RESULTS_DIR}\n")
    
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(AVG_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    all_run_metrics = []
    
    # Run experiments
    for run_id in range(NUM_RUNS):
        run_metrics = run_experiment_with_budgets(run_id)
        if run_metrics:
            all_run_metrics.append(run_metrics)
        
        # Save intermediate results after each run
        if all_run_metrics:
            intermediate_avg = average_metrics(all_run_metrics)
            intermediate_file = f"{AVG_RESULTS_DIR}/averaged_metrics_after_{len(all_run_metrics)}_runs.json"
            with open(intermediate_file, 'w') as f:
                json.dump(intermediate_avg, f, indent=2)
            print(f"\nIntermediate results saved: {intermediate_file}")
    
    # Calculate final averages
    print(f"\n{'='*60}")
    print("Calculating final averaged metrics...")
    print(f"{'='*60}\n")
    
    final_averaged = average_metrics(all_run_metrics)
    
    # Save final results
    final_file = f"{AVG_RESULTS_DIR}/final_averaged_metrics.json"
    with open(final_file, 'w') as f:
        json.dump(final_averaged, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL AVERAGED METRICS SUMMARY")
    print("="*60)
    for budget in BUDGETS:
        if budget in final_averaged:
            metrics = final_averaged[budget]
            print(f"\nBudget {budget} (n={metrics.get('num_runs', 0)} runs):")
            if 'final_clean_acc' in metrics:
                acc = metrics['final_clean_acc']
                print(f"  Clean Accuracy:    {acc['mean']:.4f} ± {acc['std']:.4f} (range: {acc['min']:.4f} - {acc['max']:.4f})")
            if 'final_poison_acc' in metrics:
                acc = metrics['final_poison_acc']
                print(f"  Poison Accuracy:   {acc['mean']:.4f} ± {acc['std']:.4f} (range: {acc['min']:.4f} - {acc['max']:.4f})")
    
    print(f"\n\nFinal results saved to: {final_file}")
    print(f"All run results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

