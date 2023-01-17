# File        :   FinalMain.py (Final Project of 2022 Spring Vision Course)
# Version     :   1.0
# Description :   Final Project 
# Date:       :   May 10, 2022
# Author      :   José Carlos Luna Zamora, Erick David Ramírez Martínez (A01748326, A01748155)@tec.mx
# License     :

import numpy as np
import cv2
import os
import math
import sys
from termcolor import colored, cprint


def readImage(imagePath):
    # Loads image:
    inputImage = cv2.imread(imagePath)
    # Checks if image was successfully loaded:
    if inputImage is None:
        print("readImage>> Error: Could not load Input image.")
    return inputImage


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# Writes an PGN image:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)



baseDir = os.path.join(os.getcwd(), "TC3050-finalProject")
os.chdir(baseDir)
mpwd = os.getcwd()
path = mpwd + "\\examples\\"
path2 = mpwd + "\\datasets\\train"
trainFiles = os.listdir(path2)
#print("trainfiles: ", trainFiles)
outArray = np.zeros((30*9, 4900 + 1), dtype=np.float32)
counter = 0
for t in trainFiles:
    actualT = int(t)
    pics = os.listdir(path2 + '\\' + t)
    for img in pics:
        inputImage = readImage(path2 + "\\" + t + '\\' + img)
        #showImage("Input" + t, inputImage)
        inputImage = cv2.cvtColor(inputImage,cv2.COLOR_BGR2GRAY)
        #showImage("Input" + t, inputImage)
        _,inputImage = cv2.threshold(inputImage,0, 255,cv2.THRESH_OTSU)
        #showImage("Input" + t, inputImage)
        outArray[counter][0] = actualT
        Ddata =  inputImage.reshape(-1, 70*70).astype(np.float32)
        outArray[counter][1:] = Ddata
        counter += 1
(trSamples, attributes) = outArray.shape
trainLabels = outArray[0:trSamples, 0:1].astype(np.int32)
trainData = outArray[0:trSamples, 1:attributes].astype(np.float32)

SVM = cv2.ml.SVM_create()

# Set hyperparameters:
SVM.setKernel(cv2.ml.SVM_LINEAR)  # Sets the SVM kernel, this is a linear kernel
SVM.setType(cv2.ml.SVM_NU_SVC)  # Sets the SVM type, this is a "Smooth" Classifier
SVM.setNu(0.1)  # Sets the "smoothness" of the decision boundary, values: [0.0 - 1.0]

SVM.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 25, 1.e-01))
SVM.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)


files = os.listdir(path)
#print(files)

for elem in files:
    inputImage = readImage(path+elem)
    final = inputImage.copy()
    #showImage("INPUTIMAGE", inputImage)
    #writeImage("INPUTIMAGE", inputImage)
    # To Gray
    grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    #showImage("grayscaleImage", grayscaleImage)
    #writeImage("grayscaleImage", grayscaleImage)

    sigma = (5, 5)
    grayscaleImage = cv2.GaussianBlur(grayscaleImage, sigma, 0)
    #showImage("grayscaleImage", grayscaleImage)

    windowSize = 67
    constantValue = 8

    binaryImage = cv2.adaptiveThreshold(grayscaleImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, windowSize, constantValue)
    #showImage("Adaptive Binary", binaryImage)
    #writeImage("Adaptive Binary", binaryImage)

    windowSize = 7
    # Create the kernel:
    smallBlur = np.ones((windowSize, windowSize), dtype="float")
    # Set the kernel with averaging coefficients:
    smallBlur = (1.0/(windowSize*windowSize)) * smallBlur
    # Apply filter to image:
    # -1 infers the data type of the output image from input image:
    imageBlur = cv2.filter2D( binaryImage, -1, smallBlur )
    #showImage( "Image Blur", imageBlur )
    #writeImage( "Image Blur", imageBlur )


    kernelSize = 5  # 5
    # Set operation iterations:
    opIterations = 2
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    # Perform Erosion:
    erodeImage = cv2.morphologyEx(imageBlur, cv2.MORPH_CLOSE, morphKernel, iterations=opIterations)
    # Check out the image:
    contours, _ = cv2.findContours(erodeImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color = (0, 0, 255)
    # Loop through the contours list:
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approxAccuracy = 0.05 * perimeter
        _,_,width, height = cv2.boundingRect(c)
        area = width*height
        if perimeter > 5500 and area > 1500000 and area < 6000000:
            cv2.drawContours(final, [c], 0, color, 3)
            vertices = cv2.approxPolyDP(c, approxAccuracy, True)
            if len(vertices) == 4:
                boundRect = cv2.boundingRect(c)
                # Get the bounding Rect data:
                rectX = int(boundRect[0])
                rectY = int(boundRect[1])
                rectWidth = 630
                rectHeight = 630
                inPoints = np.zeros((4, 2), dtype="float32")
                outPoints = np.array([
                            [0, 0],  # 1
                            [rectWidth, 0],  # 2
                            [0, rectHeight],  # 3
                            [rectWidth, rectHeight]],  # 4
                            dtype="float32")
                #cv2.rectangle(final, (rectX, rectY), (rectX + rectWidth, rectY + rectHeight), color, 2)
                misVer =[]
                for v in vertices:
                    misVer.append([v[0,0], v[0,1]])
                    #print("X: ", v[0,0], "Y: ", v[0,1])
                    cv2.circle(final, (v[0,0], v[0,1]), 5, (255, 0, 0), 5)
                    cv2.putText(final, str(v + 1), (v[0,0], v[0,1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=2)
                    windowSize = 51
                    constantValue = 2
                #showImage("Verices", final)
                #writeImage("Vertices", final)
            misVer.sort(key=lambda x: math.sqrt(x[0]*x[0] + x[1]*x[1]))
            #print("Mis Ver",misVer)
            if(misVer[2][1] < misVer[1][1]):
                temp = misVer[2]
                misVer[2] = misVer[1]
                misVer[1] = temp
            inPoints = np.array(misVer).astype("float32")
            #print("InPoints",inPoints)
            # Compute the perspective transform matrix and then apply it
            H = cv2.getPerspectiveTransform(inPoints, outPoints)
            rectifiedImage = cv2.warpPerspective(grayscaleImage, H, (rectWidth, rectHeight))
            colorRecti = cv2.warpPerspective(inputImage, H, (rectWidth, rectHeight))
            #showImage("rectifiedImage", rectifiedImage)
            #writeImage("rectifiedImage", rectifiedImage)
            windowSize = 67
            constantValue = 8
            rectifiedBinary = cv2.adaptiveThreshold(rectifiedImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, windowSize, constantValue)
            #showImage("rectifiedBinary", rectifiedBinary)
            #writeImage("rectifiedBinary", rectifiedBinary)
            contours, _ = cv2.findContours(rectifiedBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                _,_,width, height = cv2.boundingRect(c)
                area = width*height
                if(area >= 200000):
                    boundRect = cv2.boundingRect(c)
                    # Get the bounding Rect data:
                    rectX = int(boundRect[0])
                    rectY = int(boundRect[1])
                    rectWidth = int(boundRect[2])
                    rectHeight = int(boundRect[3])

                    perimeter = cv2.arcLength(c, True)
                    approxAccuracy = 0.05 * perimeter
                    vertices2 = cv2.approxPolyDP(c, approxAccuracy, True)
                    losVer = []
                    for v in vertices2:
                        losVer.append([v[0,0], v[0,1]])
                        cv2.circle(colorRecti, (v[0,0], v[0,1]), 1, (255, 0, 0), 1)
                        #showImage("COLORVerti", colorRecti)

                    losVer.sort(key=lambda x: math.sqrt(x[0]*x[0] + x[1]*x[1]))
                    if(losVer[2][1] < losVer[1][1]):
                        temp = losVer[2]
                        losVer[2] = losVer[1]
                        losVer[1] = temp
                    inPoints = np.array(losVer).astype("float32")

                    rectWidth = 630
                    rectHeight = 630
                    outPoints = np.array([
                                [0, 0],  # 1
                                [rectWidth, 0],  # 2
                                [0, rectHeight],  # 3
                                [rectWidth, rectHeight]],  # 4
                                dtype="float32")
                    H = cv2.getPerspectiveTransform(inPoints, outPoints)
                    denuez = cv2.warpPerspective(colorRecti, H, (rectWidth, rectHeight))
                    denuezBin = cv2.warpPerspective(rectifiedBinary, H, (rectWidth, rectHeight))
                    #showImage("HOLA", denuez)
                    #showImage("HolaBIN", denuezBin)
                    leftCorner = (0,0)
                    cv2.floodFill(denuezBin, None, leftCorner, 0)
                    kernelSize = 3  # 5
                    # Set operation iterations:
                    opIterations = 1
                    # Get the structuring element:
                    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
                    # Perform Erosion:
                    denuezBin = cv2.morphologyEx(denuezBin, cv2.MORPH_ERODE, morphKernel, iterations=opIterations)
                    denuezCopy = denuezBin.copy()
                    resultadoss = []
                    for j in range(9):
                        for i in range(9):
                            #cv2.rectangle(denuezBin,(70*i,70*j), (70*(i+1), 70*(j+1)),255,1)
                            #showImage("Recorte", denuezBin)
                            actualImg = denuezBin[70*j:70*j+70, 70*i:70*i+70]
                            contours, _ = cv2.findContours(actualImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            #showImage("Recorte", actualImg)
                            cero = 0
                            for c in contours:
                                x,y,width, height = cv2.boundingRect(c)
                                area = width*height
                                aspectRatio = width / height
                                if area > 300:
                                    #cv2.floodFill(denuez, None, ((70*i+x+width)//2, (70*j+y+height)//2), (255,255,255))
                                    #showImage("HOLA", denuez)
                                    cv2.rectangle(denuezBin,(x+70*i,y+70*j), (70*i+x+width, 70*j+y+height),255,1)
                                    theCrop = denuezCopy[y+70*j:70*j+y+height, x+70*i:70*i+x+width]
                                    (height, width) = theCrop.shape[:2]
                                    # Compute aspect ratio:
                                    aspectRatio = width / height
                                    newHeight = int(height*1.5)
                                    newWidth = int(newHeight * aspectRatio)
                                    # Set new size:
                                    newSize = (newWidth, newHeight)
                                    theCrop = cv2.resize(theCrop, newSize, cv2.INTER_AREA)
                                    #showImage("TheCrop", theCrop)
                                    #showImage("Ayudaaa", denuezBin)
                                    blank = np.zeros((70, 70), np.uint8)
                                    originX = 35-(newWidth//2)
                                    originY = 35-(newHeight//2)
                                    blank[originY:originY+newHeight, originX:originX+newWidth] = theCrop
                                    #showImage("aPredecir", blank)
                                    #writeImage("aPredecir", blank)
                                    #writeImage("aPredecir", blank)
                                    reShaped = blank.reshape(-1,70*70).astype(np.float32)
                                    svmResult = SVM.predict(reShaped)
                                    #text = colored(str(int(svmResult[1][0][0])), 'red')
                                    text = str(int(svmResult[1][0][0]))
                                    resultadoss.append(text)
                                    cero = 1

                            if cero == 0:
                                #print("otro lado aspect:", aspectRatio)
                                resultadoss.append(0)
                            else:
                                cero= 0
                    #showImage("Indentificados", denuezBin)
                    #writeImage("Indentificados", denuezBin)

                    f = open("Result.txt", "a")
                    counter2 = 0 
                    for j in range(9):
                        for i in range(9):
                            f.write('[')
                            f.write(str(resultadoss[counter2]))
                            f.write(']')
                            print('[', end=' ')
                            print(resultadoss[counter2], end='')
                            print(']', end=' ')
                            counter2 +=1
                        f.write("\n")
                        print("\n", end='')
                    f.write("\n")
                    print("\n")
                    f.close()

            #showImage("INPUTIMAGE", inputImage)
