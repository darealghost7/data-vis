import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Data
experience = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 15, 16, 18, 20])
education = np.array([0, 0, 1, 1, 0, 2, 1, 2, 1, 2, 0, 1, 2, 0, 1])
location = np.array([0, 1, 1, 2, 0, 2, 2, 1, 1, 0, 2, 2, 1, 0, 1])
salary = np.array([48, 53, 60, 65, 68, 80, 78, 88, 90, 100, 92, 105, 108, 115, 120])

# Mapping for visualization
colors = ['red', 'green', 'blue']  # 0: Remote, 1: On-site, 2: Hybrid
markers = ['o', 's', '^']          # 0: Bachelor's, 1: Master's, 2: PhD
loc_labels = ['Remote', 'On-site', 'Hybrid']

# Scatter plot
plt.figure(figsize=(10, 6))
for i in range(len(experience)):
    plt.scatter(experience[i], salary[i],
                color=colors[location[i]],
                marker=markers[education[i]],
                s=100, label=f'{loc_labels[location[i]]}' if i == location.tolist().index(location[i]) else "")

# Fit and plot lines of best fit per location
x_vals = np.linspace(min(experience), max(experience), 100)
for loc in range(3):  # 0: Remote, 1: On-site, 2: Hybrid
    mask = location == loc
    model = LinearRegression()
    model.fit(experience[mask].reshape(-1, 1), salary[mask])
    y_pred = model.predict(x_vals.reshape(-1, 1))
    plt.plot(x_vals, y_pred, color=colors[loc], linestyle='--', label=f'{loc_labels[loc]} Fit')

plt.title("Experience vs Salary by Education and Location")
plt.xlabel("Years of Experience")
plt.ylabel("Salary (in thousands USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
