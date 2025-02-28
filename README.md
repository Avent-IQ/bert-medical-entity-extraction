
# Model Overview

This model is a fine-tuned Denoising Diffusion Probabilistic Model (DDPM) for generating images of flowers using the Oxford Flowers dataset. It builds upon the pretrained google/ddpm-cifar10-32 model and is optimized for training on a GPU.

# Model Details
```

Architecture: UNet2DModel

Noise Scheduler: DDPMScheduler

Training Data: Oxford Flowers dataset (nelorth/oxford-flowers)

Optimizer: AdamW

Learning Rate: 1e-4, adjusted using a cosine scheduler

Training Steps: 100 epochs

Batch Size: 64

Image Size: 32x32 pixels
```

# Training Configuration

The training process involves the following steps:

# Data Preprocessing:

Images resized to 32x32.

Random horizontal flipping applied for augmentation.

Normalized to the range [-1, 1].

# Noise Addition:

Random noise added to images using a linear beta schedule.

# Model Training:

The UNet model predicts the noise added to images.

The Mean Squared Error (MSE) loss is used.

The learning rate is adjusted with a cosine scheduler.

# Checkpointing:

Model checkpoints are saved every 1000 steps.

# Usage

Once trained, the model can be used for generating images of flowers. The trained model is saved as a DDPMPipeline and can be loaded for inference.

# Model Inference
  ```python

from optimum.intel.openvino import OVModelForImageGeneration

pipeline = OVModelForImageGeneration.from_pretrained("flower_diffusion_quantized", export=True)
images = pipeline(batch_size=4, num_inference_steps=50).images
images[0].show()
```
# Model Variants

FP32 Version: Standard precision model.

FP16 Version: Reduced precision for lower memory usage.

# Limitations and Considerations

Image Resolution: Trained at 32x32, which may limit the fine details.

Computational Requirements: A GPU is recommended for inference.

Dataset Bias: The model is trained solely on Oxford Flowers, so its generalization to other datasets is limited.

Quantized Model Accuracy: INT8 quantization may slightly reduce output quality but speeds up inference.

