import numpy as np
import sys

def adjustIntensity(inImage, inRange=[], outRange=[0, 1]):
    m,n= np.shape(inImage)

    if inRange == []:
        imin = inImage.min()
        imax = inImage.max()
    else:
        imin = inRange[0]
        imax = inRange[1]
    
    omin = outRange[0]
    omax = outRange[1]

    output = np.zeros((m,n), np.float32)

    for x in range(len(inImage)):
        for y in range(len(inImage[0])):
            temp = inImage[x, y]
            #print(f"(tmp:{temp}-imin:{imin})/(imax:{imax}-imin:{imin}))*(omax:{omax} - omin:{omin})+omin")
            # NORMALIZANDO nueovs max y min
            output[x, y] = (((temp - imin)/(imax-imin))*(omax - omin)+omin)

    return output

def adjustIntensity_red(inImage, inRange=[], outRange=[0, 1]):
    m,n= np.shape(inImage)

    if inRange == []:
        imin = inImage.min()
        imax = inImage.max()
    else:
        imin = inRange[0]
        imax = inRange[1]
    
    omin = outRange[0]
    omax = outRange[1]

    output = np.zeros((m,n), np.float32)

    for x in range(len(inImage)):
        for y in range(len(inImage[0])):
            temp = inImage[x, y]
            #print(f"(tmp:{temp}-imin:{imin})/(imax:{imax}-imin:{imin}))*(omax:{omax} - omin:{omin})+omin")
            # NORMALIZANDO nueovs max y min
            output[x, y] = round((((temp - imin)/(imax-imin))*(omax - omin)+omin))

    return output

def equalizeIntensity(inImage, nBins = 256):
    m,n= np.shape(inImage)
    outImage = np.zeros((m,n),np.float32)

    # Numpy histograms
    hist, _ = np.histogram(inImage, nBins)

    totalPixels = m * n
    for x in range(m):
        for y in range(n):
            idx = inImage[x, y].astype(np.uint8)
            sum = 0
            # Sumamos todos los pixeles que tienen igual o menor nvl de gris
            for i in range(idx):
                sum = sum + hist[i]
            outImage[x, y]= sum/totalPixels
    
    return outImage

def filterImage(inImage, kernel):
    m,n = np.shape(inImage)
    p,q = np.shape(kernel)

    p2 = (p//2)
    q2 = (q//2)
    # Creamos una matriz de zeros p-1 y q-1 mas grande que la imagen
    # Insertamos la imagen en el centro de esa matriz
    inBig = np.zeros((m+p-1,n+q-1), np.float32)
    inBig[p2:p2+m,q2:q2+n] = inImage
    output = np.zeros((m,n), np.float32)

    for i in range(m):
        for j in range(n):
            aux = 0
            for a in range(p):
                for b in range(q):
                    res = inBig[i+a,j+b] * kernel[a,b]
                    aux = aux + res
            output[i,j] = aux
    return output

def gaussKernel1D(sigma):
    N = 2*3*sigma + 1

    centro = N//2 + 1

    kernel = np.zeros(int(N),np.float32)
    i=0
    for z in range(-int(centro-1),int(centro-1)):
        div = np.sqrt(2*np.pi)*sigma
        num = np.exp(-float(z)**2/(2*sigma**2))
        #kernel[i] = (1/math.sqrt(2*math.pi*sigma))*math.e ** (math.pow(-z,2)/2*math.pow(sigma,2))
        kernel[i] = 1/div * num
        i+=1
        #kernel[0+z-centro] = -kernel[z]
    return kernel

def gaussianFilter(inImage, sigma):
    kernel = np.outer(gaussKernel1D(sigma), gaussKernel1D(sigma))
    outImage = filterImage(inImage, kernel)
    return outImage

def medianFilter(inImage, filterSize):
    w = len(inImage)
    h = len(inImage[0])
    outImage = np.zeros((w, h), np.float64)
    center = filterSize//2

    B = np.zeros((w+center*2, h+center*2), np.float64)
    B[center:center+w,center:center+h] = inImage

    for x in range(w):
        for y in range(h):
            outImage[x, y] = np.median(B[x:x+filterSize,y:y+filterSize])
    
    return outImage

def erode(inImage, SE, center=[]):
    w,h = np.shape(inImage)
    p,q = np.shape(SE)
    print(f"inImage shape = [{w},{h}]\n SE shape = [{p},{q}]")
    if not center:
        cx = p//2 +1 
        cy = q//2 +1
    else:
        cx = center[0]
        cy = center[1]
    
    outer = np.zeros((w+p-1,h+q-1), np.int8)
    outer[cx:cx+w, cy:cy+h] = inImage

    output = np.zeros((w,h),np.int8)

    for x in range(cx,w+cx):
        for y in range(cy,h+cy):
            temp = 1
            for i in range(p):
                for j in range(q):
                    if SE[i,j] == 1:
                        if outer[x-cx+i,y-cy+j] != 1:
                            temp = 0
                            break
                if temp == 0:
                    break
            output[x-cx,y-cy] = temp
    
    return output


def dilate(inImage, SE, center=[]):
    w,h = np.shape(inImage)
    p,q = np.shape(SE)
    if not center:
        cx = p//2 
        cy = q//2
    else:
        cx = center[0]
        cy = center[1]
    
    outer = np.zeros((w+p-1,h+q-1), np.int8)
    outer[cx:cx+w, cy:cy+h] = inImage

    output = np.zeros((w,h),np.int8)

    for x in range(cx,w+cx):
        for y in range(cy,h+cy):
            temp = 0
            for i in range(p):
                for j in range(q):
                    if SE[i,j] == 1:
                        if outer[x-cx+i,y-cy+j] == 1:
                            temp = 1
                            break
                if temp == 1:
                    break
            output[x-cx,y-cy] = temp
    
    return output

def opening(inImage, SE, center=[]):
    output = erode(inImage, SE, center)
    output = dilate(output, SE, center)
    return output

def closing (inImage, SE, center=[]):
    output = dilate(inImage, SE, center)
    output = erode(output, SE, center)
    return output

def fill(inImage, seeds, SE=[], center=[]):
    w,h = np.shape(inImage)

    if SE.size == 0:
        SE = np.matrix([[0,1,0],
                        [1,1,1],
                        [0,1,0]])
    
    p,q = np.shape(SE)
    if not center:
        center = [0,0]
        center[0] = p //2 
        center[1] = q //2 

    outImage = np.zeros((w,h), np.int8)
    
    complementary = ~inImage+2

    for c in seeds:
        x= c[0]
        y= c[1]
        outImage[x,y] = 1
        last = outImage
        run = True
        while(run):
            outImage = dilate(last, SE, center)
            outImage = outImage & complementary
            if (outImage == last).all():
                run = False
                break
            last = outImage
    outImage = inImage | outImage
    
    return outImage

"""
    3.4. Deteccion de bordes
"""
def gradientImage(inImage, operator):
    print(operator)
    match operator:
        case 'Roberts':
            kx = [[-1.0,0.0],
                  [0.0,1.0]]
            ky = [[0.0,-1.0],
                  [1.0,0.0]]
        case 'CentralDiff':
            kx = [[-1.0,0.0,1.0]]
            ky = [[-1.0],
                  [0.0],
                  [1.0]]
        case 'Prewitt':
            kx = [[-1.0,0.0,1.0],
                  [-1.0,0.0,1.0],
                  [-1.0,0.0,1.0]]
            ky = [[-1.0,-1.0,-1.0],
                  [0.0,0.0,0.0],
                  [1.0,1.0,1.0]]
        case 'Sobel':
            kx = [[-1.0,0.0,1.0],
                  [-2.0,0.0,2.0],
                  [-1.0,0.0,1.0]]
            ky = [[-1.0,-2.0,-1.0],
                  [0.0,0.0,0.0],
                  [1.0,2.0,1.0]]
        case _:
            print("FALTA OPERADOR! Ejemplo: Sobel, Prewitt...")
            sys.exit(-1)
    
    kx = np.matrix(kx)
    ky = np.matrix(ky)

    gx = filterImage(inImage, kx)
    gy = filterImage(inImage, ky)
    return gx, gy

def LoG(inImage, sigma):
    kernel = np.zeros((3,3), np.float64)
    for z in range(3):
        kernel[0, z] = -1
        kernel[2, z] = -1
    kernel[1,0] = -1
    kernel[1,1] = 8
    kernel[1,2] = -1
    smooth = gaussianFilter(inImage, sigma)
    outImage = filterImage(smooth, kernel)
    return outImage

def edgeCanny(inImage, sigma, tlow, thigh):
    w,height = np.shape(inImage)
    smooth = gaussianFilter(inImage, sigma)
    gx, gy = gradientImage(smooth, 'Sobel')
    magnitud = np.sqrt((np.power(gx, 2)+np.power(gy, 2)))
    supr_no_max = np.zeros(magnitud.shape, np.float64)
    outImage = np.zeros((w, height), np.float64)
    dir = np.arctan2(gy, gx) * 180 / np.pi
    h = [0,1]
    v = [1,0]
    diz = [-1,1]
    dd = [1,1]
    for (x, y), angulo in np.ndenumerate(dir):
        max = 0
        mx = 0
        if ((angulo < 135 and angulo > 45) or (angulo < -45 and angulo > -135)):
            ang = v
        elif ((angulo < 45 and angulo > -45) or (angulo > 135 and angulo < -135)):
            ang = h
        elif ((angulo > 0 and angulo < 90) or (angulo < -90 and angulo > -180)):            
            ang = diz
        elif ((angulo < 0  and angulo > -90) or (angulo > 90 and angulo < 180 )):
            ang = dd
        try:
            if (magnitud[x+ang[0],y+ang[1]] < magnitud[x,y]):
                max = 1
        except IndexError:
            max = 1
        try:
            if (magnitud[x-ang[0],y-ang[1]] < magnitud[x,y]):
                mx = 1
        except IndexError:
            mx = 1
        if max and mx:
            supr_no_max[x,y] = magnitud[x,y]
    
    for x in range(w):
        for y in range(height):
            if supr_no_max[x,y] < tlow:
                outImage[x,y] = 0
            elif thigh > supr_no_max[x,y] > tlow:
                outImage[x,y] = 1
            else:
                outImage[x,y] = 2
    
    #return smooth, magnitud, supr_no_max, outImage
    return outImage

if __name__ == "__main__":
    pass