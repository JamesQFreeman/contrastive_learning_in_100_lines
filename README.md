# Contrastive Learning in 100 Lines, in PyTorch
 A simple and intuitive contrastive learning implementation
 It is a minimalistic toy project to understand how algorithm work containing following elements:

 1. Contrastive learning frameworks, e.g., SimCLR, BYOL and SimSiam.
 2. A toy dataset: [ImageNet_5Cate](https://github.com/thunderInfy/imagenet-5-categories) to play with.
 3. Constrastive training code and supervised benchmark code.

 To keep the code in minimal scale, there're no multi-gpu and FP16 support. However, I hope these acceleration can be added easily if needed.

More frameworks coming later, the progress:
- [x] BYOL, Bootstrap Your Own Latent
- [ ] SimCLR
- [ ] SimSiam (In progree)
- [ ] Barlow Twins

## Install
Only PyTorch and PIL is required, install them. Then:

```bash
git clone https://github.com/JamesQFreeman/contrastive_learning_in_100_lines.git
cd contrastive_learning_in_100_lines
python train.py
```

## Usage
Take BYOL and ResNet50 as an example, the self-supervised training code should be like following. Complete code can be found at train.py
```python
resnet = models.resnet50() # Chose a model you like
learner = BYOL(net=resnet) # Setup a learner for different framework
my_dataset = ImageNet_5Class(augmentation=False, annotation=False) # Self-supervision so no annotation needed

for _ in range(n_epoch):
    for data in trainloader:
        inputs = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        loss = learner(inputs)
        loss.backward()
        optimizer.step()
        # BYOL only, do the EMA
        learner.update_moving_average()

# save the trained encoder
torch.save(learner.online_encoder.state_dict(), "encoder.pth")
```

 ## Thanks to
 1. https://github.com/lucidrains/byol-pytorch