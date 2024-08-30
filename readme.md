# Diffusion Model Toy Project (24.07 ~ 24.08)

MNIST 데이터셋을 이용하여 diffusion model을 학습시키는 토이 프로젝트를 진행했습니다. vanilla UNet에 Cross-Attention과 WideResNet을 사용하도록 수정했고, DDPM 논문에서 언급한 것 처럼 channel size를 설정하여 학습했습니다.

처음에는 DDPM을 구현하고 이후에 DDIM과 Classifier Free Guidance를 적용했습니다. 

### Training

학습코드는 `notebooks/diffusion_colab.ipynb` 파일에서 확인 할 수 있습니다.

- GPU : T4 (Google Colab)
  - 3min/epoch, ~1.5 hours
- batch size : 256
- timestep : 1000 (w\ linear scheduling)
- $p_\text{uncond}$ : 0.1
- parameters : 33M
- loss : 0.01698 

![loss](./images/loss.png)


### Result

- DDIM with $\mathrm{len}(\tau) = 10$

![result1](./images/result.png)
![result2](./images/result2.png)
![result3](./images/result3.png)



# Reference Papers

- [Denoising Diffusion Probabilistic Model](https://arxiv.org/abs/2006.11239)
- [Denoising Implicity Probabilistic Model](https://arxiv.org/abs/2010.02502)
- [Classifier Free Guidance](https://arxiv.org/abs/2207.12598)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)