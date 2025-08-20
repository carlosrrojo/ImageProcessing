# ImageProcessing

Small python module with a few implementations of typical image processing operations.

Contains:

- **AdjustIntensity**:
    Adjusts the intensity of an image.

```
Usage: 
outImage = adjustIntensity (inImage, inRange= [], outRange= [0, 1])
    inImage: Matrix MxN with input image.
    outImage: Matrix MxN with output image.
    inRange: 1x2 vector with input range of intensity levels [imin, imax]. If vector is empty, imin and imax are the minimum and maximum intensity values of the input image. 
    outRange: 1x2 vector with output range of intensity levels [omin, omax]. Default value [0, 1]. 
```


- **EqualizeIntensity**: histogram equalization.

```
Usage: 
outImage = equalizeIntenisty(inImage, nBins= 256)
    inImage: Matrix MxN with input image.
    outImage: Matrix MxN with output image.
    nBins: number of used bins in the process. Default 256.

```

- **FilterImage**: spatial filter through convolution with an arbitrary kernel.

```
Usage: 
outImage = filterImage (inImage, kernel)
    inImage: Matrix MxN with input image.
    outImage: Matrix MxN with output image.
    kernel: PxQ matrix with kernel. We assume that the center of the kernel equals [P//2+1,Q//2+1]
```

- **GaussKernel1D**: implementation of histogram equalization.

```
Usage: 
    gaussKernel1D (inImage: Matrix, inRange: [], outRange: [0, 1]): Matrix
```

- **GaussianFilter**: implementation of histogram equalization.

```
Usage: 
    gaussianFilter (inImage: Matrix, inRange: [], outRange: [0, 1]): Matrix
```

- **MedianFilter**: implementation of histogram equalization.

```
Usage: 
    medianFilter (inImage: Matrix, inRange: [], outRange: [0, 1]): Matrix
```