#!/usr/bin/env python3
"""
Script to plot average episode rewards from TensorBoard event files.
Automatically discovers all runs in results_mad and results_informarl folders.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_files(base_dirs):
    """
    Recursively find all TensorBoard event files in the given directories.

    Args:
        base_dirs: List of base directory paths to search

    Returns:
        List of tuples: (method_name, seed_number, run_number, file_path)
    """
    event_files = []

    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Warning: Directory '{base_dir}' not found, skipping...")
            continue

        # Walk through all subdirectories
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                # Look for TensorBoard event files
                if file.startswith('events.out.tfevents'):
                    file_path = os.path.join(root, file)

                    # Extract method name from base directory
                    method_name = os.path.basename(base_dir)
                    if method_name.startswith('results_'):
                        method_name = method_name.replace('results_', '')

                    # Try to extract seed number from path
                    seed_match = re.search(r'seed[_-]?(\d+)', root, re.IGNORECASE)
                    seed_num = int(seed_match.group(1)) if seed_match else None

                    # Try to extract run number from path
                    run_match = re.search(r'run(\d+)', root, re.IGNORECASE)
                    run_num = int(run_match.group(1)) if run_match else 1

                    event_files.append((method_name, seed_num, run_num, file_path))

    return event_files


def load_tensorboard_data(file_path, tag='average_episode_rewards'):
    """
    Load data from a TensorBoard event file.

    Args:
        file_path: Path to the event file
        tag: Name of the metric to extract

    Returns:
        Tuple of (steps, values) numpy arrays
    """
    try:
        # Create EventAccumulator and load the event file
        ea = EventAccumulator(file_path)
        ea.Reload()

        # Try to find the tag (try multiple common variations)
        available_tags = ea.Tags().get('scalars', [])

        # Try exact match first
        if tag in available_tags:
            target_tag = tag
        else:
            # Try to find a tag containing the key words
            matching_tags = [t for t in available_tags if 'average' in t.lower() and 'reward' in t.lower()]
            if matching_tags:
                target_tag = matching_tags[0]
            elif available_tags:
                # Use first available tag
                target_tag = available_tags[0]
                print(f"Warning: Tag '{tag}' not found in {file_path}, using '{target_tag}'")
            else:
                print(f"Warning: No scalar data found in {file_path}")
                return None, None

        # Extract the data
        events = ea.Scalars(target_tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])

        return steps, values

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def plot_results(event_files, output_file='training_results.png', smooth_window=10):
    """
    Plot all results on a single figure.

    Args:
        event_files: List of tuples (method_name, seed_num, run_num, file_path)
        output_file: Path to save the plot
        smooth_window: Window size for smoothing curves (moving average)
    """
    plt.figure(figsize=(12, 7))

    # Define colors for different methods
    method_colors = {
        'mad': 'blue',
        'informarl': 'red',
        'mappo': 'green',
    }

    # Group data by method
    data_by_method = defaultdict(list)

    for method_name, seed_num, run_num, file_path in event_files:
        print(f"Loading: {method_name}, seed {seed_num}, run {run_num}")
        steps, values = load_tensorboard_data(file_path)

        if steps is not None and values is not None and len(values) > 0:
            data_by_method[method_name].append({
                'seed': seed_num,
                'run': run_num,
                'steps': steps,
                'values': values,
                'file': file_path
            })

    if not data_by_method:
        print("Error: No data found to plot!")
        return

    # Plot each run
    for method_name, runs in data_by_method.items():
        color = method_colors.get(method_name.lower(), None)

        for i, run_data in enumerate(runs):
            steps = run_data['steps']
            values = run_data['values']
            seed = run_data['seed']
            run_num = run_data['run']

            # Apply smoothing if window > 1
            if smooth_window > 1 and len(values) >= smooth_window:
                smoothed_values = np.convolve(values,
                                              np.ones(smooth_window)/smooth_window,
                                              mode='valid')
                smoothed_steps = steps[smooth_window-1:]
            else:
                smoothed_values = values
                smoothed_steps = steps

            # Create label
            if seed is not None:
                label = f"{method_name} (seed {seed}, run {run_num})"
            else:
                label = f"{method_name} (run {run_num})"

            # Plot
            plt.plot(smoothed_steps, smoothed_values,
                    label=label, color=color, alpha=0.7, linewidth=1.5)

    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Average Episode Rewards', fontsize=12)
    plt.title('Training Performance Comparison', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Show plot
    plt.show()


def main():
    """Main function to run the plotting script."""
    # Define base directories to search
    base_dirs = ['results_mad', 'results_informarl']

    print("Searching for TensorBoard event files...")
    event_files = find_event_files(base_dirs)

    if not event_files:
        print("No event files found! Please check the directory structure.")
        print(f"Searched in: {base_dirs}")
        return

    print(f"\nFound {len(event_files)} event file(s):")
    for method, seed, run, path in event_files:
        print(f"  - {method}, seed {seed}, run {run}: {path}")

    print("\nGenerating plot...")
    plot_results(event_files, smooth_window=10)


if __name__ == '__main__':
    main()
