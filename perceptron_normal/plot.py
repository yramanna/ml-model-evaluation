import numpy as np
import matplotlib.pyplot as plt
import sys

# Define the model name (pass as an argument or modify manually)
if len(sys.argv) > 1:
    model_name = sys.argv[1]  # Take model name from command line argument
else:
    model_name = "Unknown Model"  # Default name

# Load saved accuracy data
epochs = np.arange(1, 21)

try:
    dev_accuracies = np.load("val_accuracies.npy")  # Validation accuracy per epoch
except FileNotFoundError:
    print("Error: Accuracy files not found. Run training first.")
    sys.exit(1)

# Print model name
print(f" Plotting Learning Curve for: {model_name}")

# Plot the learning curve
plt.plot(epochs, dev_accuracies, label="Devlopment Set Accuracy", marker='s', color='red')

# Labels & Title
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title(f"Learning Curve - {model_name}")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
