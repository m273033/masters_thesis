from PIL import Image
import numpy
import cv2
import os
from tqdm import tqdm

def imagetoarray(path):
    #Definition : Convert the image to a numpy array
    image = Image.open(path, 'r')
    imageArray = numpy.array(image.getdata(), numpy.uint8).reshape(image.size[1], image.size[0], len(image.getbands()))
    
    return imageArray

def predefColors():
    #TODO : Set the colorlist as the dataset
    colorlist = {
        '0' : 0,
        '1' : 10,
        '2' : 0,
        '3' : 0,
        '4' : 0,
        '5' : 3,
        '6' : 3,
        '7' : 7,
        '8' : 8,
        '9' : 3,
        '10' : 3,
        '11' : 1,
        '12' : 11,
        '13' : 2,
        '14' : 2,
        '15' : 3,
        '16' : 11,
        '17' : 5,
        '18' : 5,
        '19' : 12,
        '20' : 12,
        '21' : 9,
        '22' : 3,
        '23' : 0,
        '24' : 4,
        '25' : 10,
        '26' : 10,
        '27' : 10,
        '28' : 10,
        '29' : 10,
        '30' : 10,
        '31' : 10,
        '32' : 10,
        '33' : 10,
        '-1' : 10
    }
    return colorlist

def convertToClasses(array, colorlist):
    #Definition : Get an array with class information encoded in RED channel
    result = numpy.zeros((array.shape[0], array.shape[1], 1))
    
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            result[i,j] = numpy.array([int(colorlist[str(array[i][j][0])])])
    return result

def main():
    base = "./old"
    colorlist = predefColors()
    for dir_ in os.listdir(base):
        for file_ in tqdm(os.listdir(base + '/' + dir_)):
            imageArray = imagetoarray(base + '/' + dir_ + '/' + file_)
            imageArray = convertToClasses(imageArray, colorlist)
            cv2.imwrite('./' + dir_ + '/' + file_, imageArray)

if __name__ == "__main__":
    main()
