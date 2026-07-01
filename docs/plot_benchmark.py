#!/usr/bin/env python3
import csv
import os
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib or numpy is not installed.")
    print("Please install them using: pip3 install matplotlib numpy")
    sys.exit(1)

# Find the csv file
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(script_dir, '..'))

csv_paths = [
    os.path.join(workspace_dir, 'output', 'matmul_benchmark.csv'),
    os.path.join(workspace_dir, 'build', 'output', 'matmul_benchmark.csv')
]

csv_file = None
for path in csv_paths:
    if os.path.exists(path):
        csv_file = path
        break

if not csv_file:
    print("Could not find matmul_benchmark.csv. Make sure to run benchmarks first.")
    sys.exit(1)

data = {}
sizes = []

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name_parts = row['Name'].split(' ')
        if len(name_parts) != 2:
            continue
        strategy = name_parts[0]
        size = name_parts[1]
        
        if size not in sizes:
            sizes.append(size)
            
        if strategy not in data:
            data[strategy] = []
            
        data[strategy].append(float(row['GFLOPS']))

labels = sizes
x = np.arange(len(labels))

# Strategies we care about
strategies = ['Naive', 'Blocked', 'AVX2', 'OpenMP']
# Ensure strategies exist in data
strategies = [s for s in strategies if s in data]

if not strategies:
    print("No valid benchmark data found to plot")
    sys.exit(1)

num_strategies = len(strategies)
width = 0.8 / num_strategies

fig, ax = plt.subplots(figsize=(10, 6))

for i, strategy in enumerate(strategies):
    # Calculate offset to center the bars properly
    offset = width * i - (0.8 / 2) + (width / 2)
    ax.bar(x + offset, data[strategy], width, label=strategy)

ax.set_ylabel('GFLOPS')
ax.set_title('Matrix Multiplication Performance by Backend')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

output_path = os.path.join(script_dir, 'benchmark_chart.png')
plt.savefig(output_path)
print(f"Chart successfully saved to {output_path}")
