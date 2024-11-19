# Training Deep-Learning Models on ARC4 using Multiple GPUs

## Introduction

Training deep-learning models on multiple GPUs can be beneficial when dealing with large datasets and complex models that require substantial computational power and memory. By distributing the workload across multiple GPUs, we can significantly reduce training time, which is useful for rapid iteration and debugging of model prototypes. This parallel processing also allows for handling larger batch sizes, which can improve model accuracy and stability. Some very large deep-learning models, such as those used for natural-language processing, may even require multiple GPUs just to make training feasible.

Multi-GPU training of neural networks is an active research field and many different approaches have been proposed. The simplest and most commonly used paradigms are:

1. **Data Parallelism**: This involves splitting the training dataset into batches and distributing these batches across multiple GPUs. Loss and gradient calculations are performed independently on each GPU before being averaged and the average gradients used to update the model parameters. Data parallelism is easy to implement and allows scaling of models to large datasets. The main disadvantage is the need to synchronize gradients each time the parameters are updated. Each GPU must also hold a complete copy of the model in memory, which can be limiting for very large models.

2. **Model Parallelism**: In this paradigm, different parts of the model are distributed across multiple GPUs. Each GPU performs a subset of the model's calculations. Model parallelism allows training of very large models that cannot fit into the memory of a single GPU. This naturally avoids the memory redundancy of having multiple copies of the model in memory across GPUs. The disadvantages of model parallelism are that it is more complex to implement and manage, and it can be difficult to balance load equally across GPUs. There is also more inter-GPU communication overhead.

In order to test the multi-GPU scalability of deep-learning models on ARC4, a computer experiment was carried out using the [MNIST](https://yann.lecun.com/exdb/mnist/) dataset and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

## Multi-GPU experiment with the MNIST dataset

### The MNIST dataset

The [MNIST](https://yann.lecun.com/exdb/mnist/) (Modified National Institute of Standards and Technology) dataset is a large collection of handwritten digits commonly used for training and testing in the field of machine learning and computer vision. It consists of 70,000 grayscale images of handwritten digits from zero to nine, each image being 28x28 pixels in size. The dataset is divided into 60,000 training images and 10,000 test images. MNIST is widely used because it is simple yet challenging enough to test various algorithms, making it a standard benchmark for evaluating the performance of classification models.

### CNN architecture

To test the performance scalability with multi-GPU training on ARC4, the following Convolutional Neural Network (CNN) was trained to classify digits in the [MNIST](https://yann.lecun.com/exdb/mnist/) dataset:

```python
# Define the CNN in PyTorch
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

This is fairly standard CNN architecture and builds 128 scalar features from the images using successive convolutions and max. pooling. Dropout is used for regularization in the penultimate layer.

### Defining the `LightningModule`

Data parallelism can be implemented directly in vanilla [PyTorch](https://pytorch.org/) but an easier way is to wrap the PyTorch code with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/). When Lightning detects multiple GPUs are present it will automatically distribute training batches across GPUs. To benefit from this we must first encapsulate our training code in a `LightningModule` class:

```python
# Define the LightningModule
class LitCNN(L.LightningModule):
    def __init__(self, num_classes):
        super(LitCNN, self).__init__()
        self.cnn = CNN(num_classes)
        self.loss_fn = F.cross_entropy
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
    
    def forward(self, x):
        return self.cnn(x)
    
    def compute_loss(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y
    
    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc.update(y_hat, y)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self.compute_loss(batch)
        self.test_acc.update(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

The `LightningModule` class defines how the model is instantiated, trained, validated and tested. It specifies how training diagnostics, such as accuracy, are returned and what type of optimization is used. The idea of the `LightningModule` is to separate these aspects of the model from the hardware-specific implementation of the training algorithm. Different multi-GPU training strategies can therefore be tested for a given model architecture and configuration.

### Training strategy

The [MNIST](https://yann.lecun.com/exdb/mnist/) dataset comes with pre-defined training and testing sets. In addition, it's useful to set aside 15% of the training data for validation so that performance metrics can be monitored throughout the training process.

```python
# Download the MNIST dataset
transform = transforms.ToTensor()
train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)
test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)

# Use 15% of the training data as validation data
train_set_size = int(0.85 * len(train_set))
valid_set_size = len(train_set) - train_set_size

# Split the training set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size], generator=seed)
```

As our model is quite small and fits easily into memory, the appropriate parallelization strategy is data parallelism, which will assign different batches of training data to each GPU.

### Data throughput

When training on multiple GPUs, it is useful to parallelize the loading of data batches to prevent additional latency. PyTorch `DataLoader` objects allow the user to specify the number of workers assigned to this task. Since we have 20 CPU cores per GPU available for training, we can allocate 19 to data loading. The loss and gradient calculations will all take place on the GPU(s).

```python
# Build the DataLoaders
num_workers = 19
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=num_workers)
```

A batch size of 128 fits comfortably in the memory of each NVIDIA V100 GPU.

### Training benchmark

The CNN was trained on the [MNIST](https://yann.lecun.com/exdb/mnist/) data for 20 epochs, which was found to be sufficient to obtain a test accuracy $>0.98$. Only the training loop itself was benchmarked, so time taken for compilation of the model etc. was not recorded.

## Results

Training was carried out on a single GPU node of ARC4. Each GPU node has four NVIDIA V100 GPUs. Due to the recent retirement of ARC3, only 12 GPUs were available for use across all the public nodes of ARC4, meaning that queues were extremely long (> 12 hours for a single GPU). For this reason, training was carried out on one and two GPUs only. The following job submission script was used for the two-GPU training:

```bash
#! /bin/bash -l

# Train using PyTorch Lightning

# Job name
#$ -N train_multi

# Output files
#$ -o output_multi.log
#$ -e error_multi.log

# Run in the current directory (-cwd)
#$ -cwd

# Request some time- min 15 mins - max 48 hours
#$ -l h_rt=00:15:00

# Request GPU node
#$ -l coproc_v100=2

# Get email at start and end of the job
#$ -m be

# Now run the job
unset GOMP_CPU_AFFINITY KMP_AFFINITY
conda activate lightning-gpu
python train.py
```

The single-GPU submission script was mostly identical except for the argument `-l coproc_v100=1` to request only one GPU.

Training times were as follows:

* **Single GPU**: 145 seconds.

* **Two GPUs**: 84.5 seconds.

That represents a 72 percent increase in the training rate. We would not typically expect to see a 100 percent increase for data-parallel training due to communication overheads of using multiple GPUs, such as the need to synchronize gradients after each training batch.

At present, the long queues for GPU nodes on ARC4 mean that queueing + training time is typically longer for multi-GPU jobs. On AIRE, with its much larger complement of GPUs, it is possible that the situation will be different, although further benchmarks will be needed to confirm this.