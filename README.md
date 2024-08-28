## Acknowledgements

This code is built on  [DAT](https://github.com/zhengchen1999/DAT). Some filenames and function names have not been modified.

## Environment

[PyTorch >= 1.7](https://pytorch.org/)

## How To Test

```
python test.py -opt options/Test/test_TranMamba_light_x4.yml
```
The testing results will be saved in the ./results folder.
Due to the model save file being too large (over 25MB), it cannot be uploaded. If you have testing needs, please contact the author.

## How To Train

```
python train.py -opt options/Train/train_TranMamba_light_x4.yml
```

## Contact
If you have any question, please email lzhang2019@lzu.edu.cn to discuss with the authors.
