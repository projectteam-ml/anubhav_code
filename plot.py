import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv("training_log.csv")


# Plot using the first and second columns
x = data.iloc[:, 0]
y = data.iloc[:, 1]  

plt.figure(figsize=(10, 5))
plt.plot(x, y, marker='o')
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.title('Plot of Column 1 vs Column 2')
plt.grid(True)
plt.tight_layout()
plt.show()
