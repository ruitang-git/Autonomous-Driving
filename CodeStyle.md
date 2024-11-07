## Coding Components
- data preprocessing
- model
- loss function

## Coding Step
1. 指定训练设备
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```
