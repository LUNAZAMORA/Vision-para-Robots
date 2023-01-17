# File        :   main_forma1.py (Practice 5 method 1 of 2022 Spring Vision Course)
# Version     :   1.0
# Description :   Practice 5 method 1
# Date:       :   May 10, 2022
# Author      :   José Carlos Luna Zamora, Erick David Ramírez Martínez (A01748326, A01748155)@tec.mx
# License     :

import numpy as np
import cv2
import os

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


path = "D:\\Carlos\\VisionRobots\\Final\\TC3050-finalProject\\examples\\"
path2 = "D:\\Carlos\\VisionRobots\\Final\\"

files = os.listdir(path)
print(files)

for elem in files:
    inputImage = readImage(path+elem)
    final = inputImage.copy()
    showImage("INPUTIMAGE", inputImage)

    # To Gray
    grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    #showImage("grayscaleImage", grayscaleImage)

    # Gaussian Blur:
    #sigma = (7, 7)
    #grayscaleImage = cv2.GaussianBlur(grayscaleImage, sigma, 0)
    #showImage("grayscaleImage", grayscaleImage)

    #_, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_OTSU)
    #showImage("Binary Image", binaryImage)
    #binaryImage = 255 - binaryImage
    #showImage("Binary Image", binaryImage)

    windowSize = 99
    constantValue = 2

    binaryImage = cv2.adaptiveThreshold(grayscaleImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, windowSize, constantValue)
    #showImage("Binary Image", binaryImage)
    #binaryImage = 255 - binaryImage
    #showImage("Binary Image", binaryImage)

    # Low pass filter (vectorized:)
    windowSize = 7
    # Create the kernel:
    smallBlur = np.ones((windowSize, windowSize), dtype="float")
    # Set the kernel with averaging coefficients:
    smallBlur = (1.0/(windowSize*windowSize)) * smallBlur
    print("Low pass kernel: ")
    #print(smallBlur)
    # Apply filter to image:
    # -1 infers the data type of the output image from input image:
    imageBlur = cv2.filter2D( binaryImage, -1, smallBlur )
    #showImage( "Image Blur", imageBlur )


    #Apply Morphology:
    # an Erosion:
    # Set kernel (structuring element) size:
    kernelSize = 15  # 5
    # Set operation iterations:
    opIterations = 1
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    # Perform Erosion:
    erodeImage = cv2.morphologyEx(imageBlur, cv2.MORPH_OPEN, morphKernel, iterations=opIterations)
    # Check out the image:
    #showImage("Erosion", erodeImage)



    contours, _ = cv2.findContours(erodeImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color = (0, 0, 255)
    #print(contours)
    # Loop through the contours list:
    #print(contours)
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approxAccuracy = 0.05 * perimeter
        _,_,width, height = cv2.boundingRect(c)
        area = width*height
        if perimeter > 5500 and area > 1500000:
            print("Perimetro", perimeter)
            print("Area: ", area)
            cv2.drawContours(final, [c], 0, color, 3)
            #showImage("FINALLL", final)
            vertices = cv2.approxPolyDP(c, approxAccuracy, True)
            #print("SHAPE: ", vertices[0].shape)
            if len(vertices) == 4:
                #print(vertices)

                boundRect = cv2.boundingRect(c)
                # Get the bounding Rect data:
                rectX = int(boundRect[0])
                rectY = int(boundRect[1])
                rectWidth = int(boundRect[2])
                rectHeight = int(boundRect[3])

                cv2.rectangle(final, (rectX, rectY), (rectX + rectWidth, rectY + rectHeight), color, 2)
                #inPoints = np.zeros((4, 2), dtype="float32")
                for v in vertices:
                    cv2.circle(final, (v[0,0], v[0,1]), 5, (255, 0, 0), 5)
                    cv2.putText(final, str(v + 1), (v[0,0], v[0,1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=2)
                    #showImage("FINALLL", final)
                    newFinal = grayscaleImage[rectY:rectY+rectHeight, rectX:rectX+rectWidth]
                    #thresh, newBin = cv2.threshold(newFinal, 180, 255, cv2.THRESH_BINARY_INV)
                    #print("OTSU: ", thresh)
                    windowSize = 51
                    constantValue = 2

                    binaryImage = cv2.adaptiveThreshold(newFinal, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, windowSize, constantValue)
                    #showImage("Binary Image", binaryImage)
                showImage("FINALLL", final)
                showImage("Binary Image", binaryImage)
    #writeImage(elem + "FINAL", final)