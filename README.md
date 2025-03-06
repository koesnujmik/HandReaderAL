# HandReader Advanced Techniques for Efficient Fingerspelling Recognition

We introduce three architecture each model possesses distinct advantages and
achieves state-of-the-art results on the ChicagoFSWild and
ChicagoFSWild+ datasets.
For more information see our arxiv paper TBA.

## Installation

Clone and install required python packages:

1. Create env for RGB and KP+RGB
   - create conda env with `conda create -n RGB_KP python=3.9`
   - install torch with `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`
   - install turbojpeg with `conda install conda-forge::pyturbojpeg=1.7.7`
   - install other dependicies with `pip install -r RGB_KP_reqs.txt`

2. Create env for KP
   - create conda env with `conda create -n KP python=3.9`
   - install torch with `pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118`
   - install turbojpeg with `conda install conda-forge::pyturbojpeg=1.7.7`
   - install other dependicies with `pip install -r KP_reqs.txt`

## Dataset

TBA

## Models


| Recoginzers         | ChicagoFSWild | ChicagoFSWild+ | Znaki|
|---------------------|----------|----------|---------|
| HandReader_{KP}  | 69.3     |72.4  |92.65 |
| HandReader_{RGB}  | 72.0     | 73.8 |92.39 |
| HandReader_{RGB_KP}  | 72.9 |75.6| 94.94|

We provide models for each architecture trained on datasets ChicagoFSWild, ChicagoFSWild+, and Znaki, which can be downloaded from the link below.

| Pretrained models       | Link |
|---------------------|----------|
| HandReader_ChicagoFSWild  | [download](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/znaki/HandReader/HandReader_ChicagoFSWild.zip) |
| HandReader_ChicagoFSWild+  | [download](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/znaki/HandReader/HandReader_ChicagoFSWild%2B.zip) |
| HandReader_Znaki  | [download](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/znaki/HandReader/HandReader_Znaki.zip) |


<h2>Train</h2>
You can use downloaded trained models, otherwise select a parameters for training in `configs` folder.
To start train select config for znaki dataset, *_znaki.yaml can be used, specified all needed paths, do the same for *_chicago.yaml. Then run:

```bash
python src/train.py --config-name <cgf_name.yaml>`
```

example to start training model for kp_rgb with dataset Znaki:

```bash
python src/train.py --config-name kp_rgb_Znaki.yaml`
```


<h2>Test</h2>
To test model with provided weights. * - could be either KP, RGB, KP_RGB based on which model type was trained. For example above:

```bash
python src/test_KP_RGB.py --config-name kp_rgb_Znaki.yaml
```

## Demo

TBA

## Authors and Credits

- [Korotaev Pavel](https://www.linkedin.com/in/pavel-korotaev-332406211/)
- [Surovtsev Petr](https://www.linkedin.com/in/petros000)
- [Kapitanov Alexander](https://www.linkedin.com/in/hukenovs)
- [Kvanchiani Karina](https://www.linkedin.com/in/kvanchiani)
- [Nagaev Alexander](https://www.linkedin.com/in/nagadit/)

<!-- ## Links

- [Github]()
- [arXiv]()
- [Habr]()
- [Paperswithcode]() -->
