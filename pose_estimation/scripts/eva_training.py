import numpy as np
import matplotlib.pyplot as plt

load = "results/simple_architecture/lr_0.1/in_224x224/new_5_epochs"
safe = "results/simple_architecture/lr_0.1/in_224x224/new_5_epochs"

train_loss = np.load(f"{load}/training_loss.npy") 
val_loss = np.load(f"{load}/validation_loss.npy") 

train_MSE = np.load(f"{load}/training_MSE.npy")
val_MSE = np.load(f"{load}/validation_MSE.npy")



# loss

# train
plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', linestyle='-', color='red', label='Training Set')
# validation
plt.plot(range(1, len(val_loss) + 1), val_loss, marker='o', linestyle='-', color='blue', label='Validation Set')

plt.title('Cross Entropy Loss Across Training Epochs')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy')
plt.legend()
plt.grid(True)
plt.savefig(f"{safe}/training_loss_per_epoch.png")
plt.close()

# MSE

# train
plt.plot(range(1, len(train_MSE) + 1), train_MSE, marker='o', linestyle='-', color='red', label='Training Set')
# validation
plt.plot(range(1, len(val_MSE) + 1), val_MSE, marker='o', linestyle='-', color='blue', label='Validation Set')

plt.title('Mean Squared Error Across Training Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.savefig(f"{safe}/training_MSE_per_epoch.png")
plt.close()