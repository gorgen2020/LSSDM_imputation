import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

with open('generated_outputs_nsample100.pk', "rb") as f:
    data = pickle.load(f)

all_generated_samples, all_target, all_evalpoint, all_observed_point, all_observed_time, scaler, mean_scaler = data

sample = all_generated_samples.median(dim=1)


sample = sample.values.cpu().numpy()
all_target1 = all_target.cpu().numpy()
all_evalpoint1 = all_evalpoint.cpu().numpy()

sample = sample * all_evalpoint1

# Calculate mean
mean = np.mean(sample)

# Calculate variance
variance = np.var(sample)

print(f"Mean: {mean}")
print(f"Variance: {variance}")


