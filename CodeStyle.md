## Coding Components
- data preprocessing
- model
- loss function

## Coding Step
1. 指定训练设备
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
