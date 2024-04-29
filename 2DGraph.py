import pandas as pd
import matplotlib.pyplot as plt

# Read the csv file
df = pd.read_csv(r"C:\Users\LEGION\PycharmProjects\pythonProject\Airlines.csv")

# Plot a bar graph for market price and quantity
plt.figure(figsize=(10,6))
plt.bar(df['Flight'], df['Length'])
plt.xlabel('Flightt')
plt.ylabel('Length')
plt.title('flight and length')
plt.show()