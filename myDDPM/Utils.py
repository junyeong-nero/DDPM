from matplotlib import pyplot as plt

def print_image(image):
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.show()
    
def print_loss(loss_values):
    epochs = list(range(1, len(loss_values) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_values, 'b-o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.legend()
    plt.grid(True)
    plt.show()