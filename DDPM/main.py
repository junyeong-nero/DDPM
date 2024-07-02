import torch 
from Encoder import ForwardEncoder
from datasets import load_dataset
from DDPM import DDPM, NoiseSchedule
import Utils

from torchvision import transforms

def extract_sample_image(index=0):
    data = load_dataset("ylecun/mnist")
    data = data.with_format("torch")
    image = data['train'][index]['image']
    Utils.print_image(image)
    return image

def T_noise():
    TIME_STEPS = 1000
    noise_schedule = NoiseSchedule(num_timesteps=TIME_STEPS)
    encoder = ForwardEncoder(noise_schedule=noise_schedule)
    image = torch.load('sample_image.pt')
    # print_image(image)
    for i in range(0, TIME_STEPS, 100):
        noised_image, epsilon = encoder.noise(image, i)
        print(i)
        Utils.print_image(noised_image)
        # print_image(epsilon)
        
def add_padding(tensor, padding_size, padding_value=0):
    if tensor.ndim != 2:
        raise ValueError("입력 텐서는 2차원이어야 합니다.")
    
    # 새로운 텐서의 크기 계산
    original_height, original_width = tensor.shape
    new_height = original_height + 2 * padding_size
    new_width = original_width + 2 * padding_size
    
    # 새로운 텐서를 패딩 값으로 초기화
    padded_tensor = torch.full((new_height, new_width), padding_value)
    
    # 원래 텐서의 값을 패딩된 텐서의 중앙에 복사
    padded_tensor[padding_size:padding_size + original_height, padding_size:padding_size + original_width] = tensor
    
    return padded_tensor
        
def T_train():
    def transform_images(batch):
        batch['image'] = add_padding(batch['image'], 2)
        return batch
    
    dataset = load_dataset("ylecun/mnist").with_format("torch")
    train, test = dataset['train'], dataset['test']
    
    train = train.map(transform_images, batched=False)
    Utils.print_image(train[0]['image'])
    
    # D = DDPM(n_timesteps=1000, train_set=train)
    # D.train()
    
    
if __name__ == '__main__':
    # extract_sample_image()
    # T_noise()
    T_train()
    