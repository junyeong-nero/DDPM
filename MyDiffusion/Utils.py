from matplotlib import pyplot as plt
import torch

def image_normalize(image):
    image = image.cpu()
    n_channels = image.shape[0]
    for channel in range(n_channels):
        max_value = torch.max(image[channel])
        min_value = torch.min(image[channel])
        image[channel] = (image[channel] - min_value) / (max_value - min_value)

    image = image.permute(1, 2, 0)

    return image

def print_image(image):
    image = image_normalize(image)
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.show()

def print_2images(image1, image2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_normalize(image1))
    axes[0].set_title('Image 1')

    axes[1].imshow(image_normalize(image2))
    axes[1].set_title('Image 2')

    plt.tight_layout()
    plt.show()

def print_digits(result):
    fig, axes = plt.subplots(1, 10, figsize=(10, 5))

    B = result.shape[0]
    for i in range(B):
        axes[i].imshow(image_normalize(result[i]))
        axes[i].set_title(i)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def print_result(result):
    for original_image, noised_image, denoised_image in result:
        batch_size = original_image.shape[0]
        for idx in range(batch_size):
            print_2images(original_image[idx], denoised_image[idx])
            # print_image(image[idx])
            # print_image(noised_image[idx])
            # print_image(denoised_image[idx])


def print_seq(loss_values,
               label='Training Loss',
               x_label='Epoch',
               y_label='Loss',
               title='Loss vs. Epochs'):
    epochs = list(range(1, len(loss_values) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_values, 'b-o', label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()