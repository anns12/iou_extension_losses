This repo implement iou, g_iou and d_iou loss classes for pytorch training

It is based on the code from this [repo](https://github.com/lilanxiao/Rotated_IoU)

### Installation

Installing all the requirement packages
```
pip install -r requirements.txt
```

Compiling the CUDA extension

```
cd Rotated_IoU/cuda_op
python setup.py install
```

Running a test
```
cd ../..
python test.py
```

**Note**: This module is tested with pytorch 1.4 and CUDA version 10.2