# GPU in the Cloud | cnn-benchmarks

Benchmark for GPUs available in the Computing Clouds using popular Convolutional Neural Network models. We use desktop GTX 1080 and CPU for the reference.

This benchmarks is based on [jcjohnson/cnn-benchmarks](https://github.com/jcjohnson/cnn-benchmarks).

We use the following GPUs for benchmarking:

|GPU|Cloud|Instance Name|Memory|Architecture|CUDA Cores|FP32 TFLOPS|Release Date|
|---|---|---|---|---|---:|---:|---|
|[Quadro P5000](https://www.techpowerup.com/gpudb/2864/quadro-p5000)|[Paperspace](https://www.paperspace.com)|P5000|16GB GDDRX5|Pascal|2560|8.22|Oct 2016|
|[Quadro M4000](https://www.techpowerup.com/gpudb/2757/quadro-m4000)|[Paperspace](https://www.paperspace.com)|GPU+|8GB GDDR5|Maxwell|1664|2.57|Jun 2015|
|[Tesla K80](https://www.techpowerup.com/gpudb/2616/tesla-k80m)|[Amazon EC2](https://aws.amazon.com), [Azure](http://azure.microsoft.com/), [Google Cloud](https://cloud.google.com)|p2, NCx, K80|12GB GDDR5|Kepler|2496|2.80|Nov 2014|
|[GRID K520](https://www.techpowerup.com/gpudb/2312/grid-k520)|[Amazon EC2](https://aws.amazon.com)|g2|4GB GDDR5|Kepler|1536|2.45|Jul 2013|
|[GTX 1080](http://www.geforce.com/hardware/10series/geforce-gtx-1080)|Desktop||8GB GDDRX5|Pascal|2560|8.87|May 2016|

Some general conclusions from this benchmarking:
- **Quadro P5000 == GTX 1080**: Performance of both GPUs is very close on all models. The main difference is twice more memory in server-side Quadro P5000.
- **Quadro P5000 > Quadro M40000**: Across all models, the Quadro P5000 is **2.2x to 2.5x** faster.
- **Quadro M4000 > Tesla K80**: Across all models, the Quadro M4000 is **1.05x to 1.25x** faster than Tesla K80, but has less memory.
- **Tesla K80 > GRID K520**: Across all models, the Tesla K80 is **1.8 to 2.25** faster than GRID K520.
- **Prefer latest cuDNN**: cuDNN5.1.10 is slightly faster than 5.1.05 which in turn is faster than 5.0.05. However at least one caveat was noticed with cuDNN5.1.10 - 8GB GTX 1080 failed on [ResNet-152](#resnet-152) while previous cuDNN versions run the model fine.

All benchmarks were run in Torch, Ubuntu 14.04 with the CUDA 8.0 Release Candidate.

All settings and models are exactly the same as in the [jcjohnson/cnn-benchmarks](https://github.com/jcjohnson/cnn-benchmarks).



## AlexNet
(input 16 x 3 x 224 x 224)

We use the [BVLC AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) from Caffe.

AlexNet uses grouped convolutions; this was a strategy to allow model parallelism over two GTX 580
GPUs, which had only 3GB of memory each. Grouped convolutions are no longer commonly used, and are
not even implemented by the [torch/nn](https://github.com/torch/nn) backend; therefore we can only
benchmark AlexNet using cuDNN.

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|Quadro P5000             |5.1.10 |   5.91|  13.68|  19.58|
|GTX 1080                 |5.1.10 |   5.97|  13.87|  19.84|
|GTX 1080                 |5.1.05 |   7.00|  13.74|  20.74|
|GTX 1080                 |5.0.05 |   7.37|  15.80|  23.16|
|Quadro M4000             |5.1.05 |  14.23|  29.52|  43.75|
|Tesla K80                |5.1.10 |  15.98|  31.63|  47.61|
|GRID K520                |5.1.10 |  39.77|  66.51| 106.28|


## Inception-V1
(input 16 x 3 x 224 x 224)

We use the Torch implementation of Inception-V1 from
[soumith/inception.torch](https://github.com/soumith/inception.torch).

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|GTX 1080                 |5.1.10 |  15.79|  35.99|  51.78|
|Quadro P5000             |5.1.10 |  16.03|  36.83|  52.86|
|GTX 1080                 |5.0.05 |  16.08|  40.08|  56.16|
|Quadro M4000             |5.1.05 |  40.29|  89.48| 129.77|
|Tesla K80                |5.1.10 |  45.43| 111.21| 156.64|
|GRID K520                |5.1.10 |  86.28| 226.87| 313.15|


## VGG-16
(input 16 x 3 x 224 x 224)

This is Model D in [[3]](#vgg-paper) used in the ILSVRC-2014 competition,
[available here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md).

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|Quadro P5000             |5.1.10 |  58.16| 122.14| 180.30|
|GTX 1080                 |5.1.05 |  59.37| 123.42| 182.79|
|GTX 1080                 |5.1.10 |  60.27| 123.15| 183.42|
|GTX 1080                 |5.0.05 |  67.27| 166.17| 233.43|
|Quadro M4000             |5.1.05 | 144.84| 299.51| 444.35|
|Tesla K80                |5.1.10 | 153.67| 295.74| 449.40|
|CPU: Dual Xeon E5-2630 v3|None   |3101.76|5393.72|8495.48|


## VGG-19
(input 16 x 3 x 224 x 224)

This is Model E in [[3]](#vgg-paper) used in the ILSVRC-2014 competition,
[available here](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md).


|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|Quadro P5000             |5.1.10 |  67.68| 139.79| 207.47|
|GTX 1080                 |5.1.10 |  69.27| 140.89| 210.16|
|GTX 1080                 |5.1.05 |  68.95| 141.44| 210.39|
|GTX 1080                 |5.0.05 |  79.79| 202.02| 281.81|
|Quadro M4000             |5.1.05 | 169.70| 347.80| 517.50|
|Tesla K80                |5.1.10 | 179.85| 347.85| 527.69|
|CPU: Dual Xeon E5-2630 v3|None   |3609.78|6239.45|9849.23|



## ResNet-18
(input 16 x 3 x 224 x 224)

This is the 18-layer model described in [[4]](#resnet-cvpr) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|GTX 1080                 |5.1.10 |  14.48|  29.35|  43.83|
|GTX 1080                 |5.1.05 |  14.62|  29.32|  43.94|
|Quadro P5000             |5.1.10 |  14.58|  29.48|  44.06|
|GTX 1080                 |5.0.05 |  14.84|  32.68|  47.52|
|Quadro M4000             |5.1.05 |  35.13|  74.08| 109.21|
|Tesla K80                |5.1.10 |  37.87|  74.88| 112.74|
|GRID K520                |5.1.10 |  64.82| 140.53| 205.36|
|CPU: Dual Xeon E5-2630 v3|None   | 859.70|1360.30|2220.01|


## ResNet-34
(input 16 x 3 x 224 x 224)

This is the 34-layer model described in [[4]](#resnet-cvpr) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|GTX 1080                 |5.1.05 |  24.50|  47.59|  72.09|
|GTX 1080                 |5.1.10 |  24.31|  47.86|  72.17|
|Quadro P5000             |5.1.10 |  24.57|  48.04|  72.61|
|GTX 1080                 |5.0.05 |  24.76|  55.00|  79.76|
|Quadro M4000             |5.1.05 |  59.09| 118.13| 177.22|
|Tesla K80                |5.1.10 |  64.79| 124.24| 189.03|
|GRID K520                |5.1.10 | 112.04| 231.02| 343.06|
|CPU: Dual Xeon E5-2630 v3|None   |1530.01|2435.20|3965.21|


## ResNet-50
(input 16 x 3 x 224 x 224)

This is the 50-layer model described in [[4]](#resnet-cvpr) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|Quadro P5000             |5.1.10 |  48.77|  98.72| 147.49|
|GTX 1080                 |5.1.10 |  50.24|  98.41| 148.65|
|GTX 1080                 |5.1.05 |  50.64|  99.18| 149.82|
|GTX 1080                 |5.0.05 |  50.76| 103.35| 154.11|
|Quadro M4000             |5.1.05 | 117.52| 228.17| 345.69|
|Tesla K80                |5.1.10 | 124.38| 274.43| 398.81|
|CPU: Dual Xeon E5-2630 v3|None   |2477.61|4149.64|6627.25|


## ResNet-101
(input 16 x 3 x 224 x 224)

This is the 101-layer model described in [[4]](#resnet-cvpr) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|Quadro P5000             |5.1.10 |  75.21| 148.67| 223.88|
|GTX 1080                 |5.1.10 |  76.92| 147.43| 224.34|
|GTX 1080                 |5.1.05 |  77.59| 148.21| 225.80|
|GTX 1080                 |5.0.05 |  77.39| 158.19| 235.58|
|Quadro M4000             |5.1.05 | 186.16| 350.82| 536.98|
|Tesla K80                |5.1.10 | 199.41| 486.11| 685.52|
|CPU: Dual Xeon E5-2630 v3|None   |4414.91|6891.33|11306.24|



## ResNet-152
(input 16 x 3 x 224 x 224)

This is the 152-layer model described in [[4]](#resnet-cvpr) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

Curiously cuDNN5.1.10 on the 8GB GTX 1080 run out of memory while previous versions of cuDNN managed to run the model fine.

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|Quadro P5000             |5.1.10 | 106.26| 204.86| 311.13|
|GTX 1080                 |5.1.05 | 109.32| 204.98| 314.30|
|GTX 1080                 |5.0.05 | 109.64| 218.62| 328.26|
|Quadro M4000             |5.1.05 | 264.14| 482.02| 746.16|
|Tesla K80                |5.1.10 | 283.68| 700.15| 983.83|
|CPU: Dual Xeon E5-2630 v3|None   |6572.17|10300.61|16872.78|


## ResNet-200
(input 16 x 3 x 224 x 224)

This is the 200-layer model described in [[5]](#resnet-eccv) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

Even with a batch size of 16, the 8GB GTX 1080, M4000 and K520 did not have enough memory to run
the model.

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|Quadro P5000             |5.1.10 | 146.78| 275.36| 422.14|
|Tesla K80                |5.1.10 | 385.33| 904.29|1289.63|
|CPU: Dual Xeon E5-2630 v3|None   |8666.43|13758.73|22425.16|


## Citations

<a id='alexnet-paper'>
[1] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. "ImageNet Classification with Deep Convolutional Neural Networks." NIPS 2012

<a id='inception-v1-paper'>
[2] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
Dragomir Anguelov, Dumitru Erhan, Andrew Rabinovich.
"Going Deeper with Convolutions." CVPR 2015.

<a id='vgg-paper'>
[3] Karen Simonyan and Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition." ICLR 2015

<a id='resnet-cvpr'>
[4] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep Residual Learning for Image Recognition." CVPR 2016.

<a id='resnet-eccv'>
[5] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Identity Mappings in Deep Residual Networks." ECCV 2016.
