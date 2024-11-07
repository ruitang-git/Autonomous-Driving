## Coding Components
- data preprocessing
- model
- loss function

## Coding Step 👉[pytorch official reference](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
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
     - a. 决定是否要加载存储系数
        ```python
        ## 一般只加载model参数
        
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        ```
        
     - b. 决定模型训练方式（部分or全部）
       ```python
       for param in model.backbone.parameters():
         param.requires_grad = False
       ```
     - c. 选择损失函数
       ```python
       loss_fn = torch.nn.CrossEntropyLoss()
       ```
     - d. 选择优化器
       ```python
       params = [p for p in model.parameters() if p.requires_grad]
       optimizer = torch.optim.SGD(params, lr = 0.0001)
       ```
     - e. 选择学习率控制器(optional)
       ```python
       lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)
       ```
     - f. 训练
      建议新建一个utils文件夹，将训练脚本放在该文件夹下
      ```python
      mse = utils.train_one_epoch(model, optimizer, train_data_loader, device)
      lr_scheduler.step()  ## 在训练脚本和测试脚本之间插入lr控制器
      
      def train_one_epoch(...):
         model.train()
         for i, data in enumerate(training_loader):
            inputs, label = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
      
      ```
      - g. 测试
        同样建议将测试脚本放在utils文件夹下
        ```python
        mse = utils.evaluate(model, val_data_loader, device)

        loss_all = 0
        def evaluate(...):
           model.eval()
           with torh.no_grad():
              for i, data in enumerate(val_data_loader):
                 inputs, label = data
                 outputs = model(inputs)
                 loss = loss_fn(outputs, label)
                 loss_all += loss
        loss_mean = loss_all/(i+1)
        ```
      - h. 保存模型
        以下按每间隔指定数量的epoch存储进行演示
        ```python

        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}

        torch.save(save_files, "./save_weights/model-{}.pth".format(epoch))
        ```

      - overall procedure
       👉[pytorch official reference](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)


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
