## Coding Components
- data preprocessing
- model
- loss function

## Coding Step
1. 指定训练设备
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```
2. 数据预处理
   a. 数据变换，数据增强
   ```python
    import torchvision.transforms as transforms
    from PIL import Image
    
    # 定义一系列数据增强和预处理操作
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为 224x224
        transforms.RandomHorizontalFlip(0.5),  # 以 0.5 的概率随机水平翻转
        transforms.ToTensor(),  # 将 PIL 图像转换为 PyTorch 张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
    ])
    
    # 加载一张图像
    image_path = "your_image.jpg"
    image = Image.open(image_path)
    
    # 应用数据增强和预处理操作
    transformed_image = transform(image)

   ```
   b. 创建DataSet数据类型
   c. 创建DataLoader类型
   ```python
    # 下述给出代码同时完成以上三步的操作（流程化操作，一定要背下来！！！）
   
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset, DataLoader
    
    # 定义数据增强和预处理操作
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 假设这是一个自定义的图像数据集类
    class CustomImageDataset(Dataset):
        def __init__(self, data_paths, labels, transform=None):
            self.data_paths = data_paths
            self.labels = labels
            self.transform = transform
    
        def __len__(self):
            return len(self.data_paths)
    
        def __getitem__(self, idx):
            image_path = self.data_paths[idx]
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            return image, label
    
    # 示例数据路径和标签
    data_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    labels = [0, 1, 0]
    
    # 创建数据集和数据加载器
    dataset = CustomImageDataset(data_paths, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 遍历数据加载器
    for images, labels in dataloader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels: {labels}")
   ```
3. 搭建模型
建议新建一个文件夹名为”network“，在其中新建.py文件书写关于模型的代码
```python
model = create_model(...)
model.to(device)
```
4. 训练，测试（含损失函数定义）
     - a. 决定模型训练方式（部分or全部）
       ```python
       for param in model.backbone.parameters():
         param.requires_grad = False
       ```
     - b. 选择优化器
       ```python
       params = [p for p in model.parameters() if p.requires_grad]
       optimizer = torch.optim.SGD(params, lr = 0.0001)
       ```



* Others:
1. 使用argparse进行传参
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu', help='device')
arg = parser.parse_args()
main(args)

def main(args):
  device = torch.device(args.device)
```
