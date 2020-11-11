import cv2,os
import numpy as np
from PIL import Image
import skimage.color

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()

modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        # my_img = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
        PIL_img = Image.open(imagePath) # grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        h, w = img_numpy.shape[:2]
        # print(len(img_numpy.shape))
        # img_numpy = skimage.color.gray2rgb(img_numpy)
        # img_numpy = np.array([[[s,s,s] for s in r] for r in img_numpy],dtype="u1")
        # img2 = cv2.merge((img_numpy,img_numpy,img_numpy))
        # print(len(img_numpy.shape))
        gray = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
        img2 = np.zeros_like(img_numpy)
        # print(len(gray.shape))
        img2[:,:,0] = gray
        img2[:,:,1] = gray
        img2[:,:,2] = gray
        blob = cv2.dnn.blobFromImages(cv2.resize(img2, (300, 300,)), 1.0, (300, 300), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces = net.forward()
        for i in range(faces.shape[2]):
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faceSamples.append(img_numpy[y:y1+10,x:x1+10])
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))