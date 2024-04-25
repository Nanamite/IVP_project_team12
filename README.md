# denoising_autoencoder

---------------
This is a part of course project for Image and Video Processing (EE6310).
Team 12
---------------

This is a simple autoencoder based implementation for denoising images. For training, the dataset used in (DnCNN-Pytorch)[https://github.com/SaoYan/DnCNN-PyTorch] is used here too. Testing is done on Set12 and Set68 respectively.

Everything can be visualized in `visualize.ipynb`. For training again, the train function can be utilized from `utils.py`.

## infer results
`infer.py` can be used to denoise a noisy image. A clean image can be provided, on which a noise with level 15 is added before denoising. If passing a noisy image use the flag `--alredy_noisy`. for general infering:
```
python infer.py --img_path <img_path> --save_dir <save directory>
```
