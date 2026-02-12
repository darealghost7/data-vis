# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 11:34:59 2025

@author: darylTshitenge
"""

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of participants and days
num_participants = 15
num_days = 14

# Step 1: Randomly assign fitness levels (0 = Low, 1 = Moderate, 2 = High)
fitness_levels = np.random.choice([0, 1, 2], size=num_participants)

# Step 2: Define means and standard deviations for each fitness level
fitness_params = {
    0: {'mean': 6000, 'std': 600},
    1: {'mean': 7500, 'std': 500},
    2: {'mean': 9000, 'std': 700}
}

# Step 2-4: Generate step counts based on fitness level
step_counts = np.zeros((num_participants, num_days), dtype=int)

for i, level in enumerate(fitness_levels):
    mean = fitness_params[level]['mean']
    std = fitness_params[level]['std']
    # Generate step counts and apply rounding and clipping
    steps = np.random.normal(loc=mean, scale=std, size=num_days)
    steps = np.round(steps).astype(int)
    steps = np.clip(steps, 3000, 15000)
    step_counts[i] = steps

# Convert fitness levels to a DataFrame for display
fitness_levels_df = pd.DataFrame({
    'Participant': np.arange(1, num_participants + 1),
    'Fitness_Level': fitness_levels
})

# Convert step counts to DataFrame
step_counts_df = pd.DataFrame(
    step_counts,
    index=[f'P{i+1}' for i in range(num_participants)],
    columns=[f'Day_{j+1}' for j in range(num_days)]
)

# Display fitness levels and step counts
print("Fitness Levels Assigned to Each Participant:")
print(fitness_levels_df.to_string(index=False))

print("\nGenerated Step Count Dataset (Steps per Day):")
print(step_counts_df)

# ----------------------------
# PART 2: DATA ANALYSIS
# ----------------------------

# (a) Average daily steps per participant
avg_daily_steps = step_counts_df.mean(axis=1)
avg_sorted = avg_daily_steps.sort_values(ascending=False)

print("\n(a) Top 5 Participants by Average Daily Steps:")
top5 = avg_sorted.head(5)
for participant, avg in top5.items():
    print(f"{participant}: {avg:.0f} steps")

# (b) Overall mean and standard deviation (rounded)
overall_mean = np.round(step_counts_df.values.mean())
overall_std = np.round(step_counts_df.values.std())

print(f"\n(b) Overall Mean of All Steps: {overall_mean:.0f}")
print(f"Overall Standard Deviation of All Steps: {overall_std:.0f}")

# (c) Median daily steps per participant
median_daily_steps = step_counts_df.median(axis=1)
highest_median = median_daily_steps.max()
lowest_median = median_daily_steps.min()

highest_participant = median_daily_steps[median_daily_steps == highest_median].index.tolist()
lowest_participant = median_daily_steps[median_daily_steps == lowest_median].index.tolist()

print("\n(c) Participant(s) with Highest Median Daily Steps:")
for p in highest_participant:
    print(f"{p}: {highest_median:.0f} steps")

print("\nParticipant(s) with Lowest Median Daily Steps:")
for p in lowest_participant:
    print(f"{p}: {lowest_median:.0f} steps")

# (d) Count how many participants have an average above 8000
above_8000 = (avg_daily_steps > 8000).sum()
print(f"\n(d) Number of Participants with Average Daily Steps > 8000: {above_8000}")

# (e) Percentiles (25th, 50th, 75th)
percentiles = np.percentile(step_counts_df.values, [25, 50, 75])
print("\n(e) Step Count Percentiles (All Data Combined):")
print(f"25th Percentile: {percentiles[0]:.0f}")
print(f"50th Percentile (Median): {percentiles[1]:.0f}")
print(f"75th Percentile: {percentiles[2]:.0f}")

