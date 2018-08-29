import cv2
import sys
import logging as log
import datetime as dt
import time
import math
import threading
import queue
from captureImage import captureImage
from statistics import stdev
from statistics import mean
from faceObject import faceObject
from random import randint

import cProfile, pstats, io
pr = cProfile.Profile()

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

anterior = 0

maxDistance = 30 # max distance a face moves per frame

dataset = []
imageStream = queue.Queue()                                             #FIFO data structure

pr.enable()                                                             #start profiler

def imageRead(q):
    while True:
        q.put(captureImage())                                           #capture image and append to queue

imageReader = threading.Thread(target=imageRead, args=(imageStream,))   #create thread for capturing images
imageReader.daemon = True                                               #thread will close when main thread exits
imageReader.start()
        
frame = imageStream.get()                                               #get one image from queue, will wait if there is no content
print(len(frame[0]))                                                    #print image size
print(len(frame))                                                       #print image size
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
initialFaces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))#initial set of faces for tracking

faceList = []
for boundingBox in initialFaces:
    randomColor = [randint(0, 255), randint(0, 255), randint(0, 255)]   #make a randomly colored rectangle around each detected face
    print([boundingBox, randomColor])
    
    faceList.append(faceObject(boundingBox, randomColor))
    
while True:
    start_time = time.time()

    # Capture frame-by-frame
    frame = imageStream.get()
    
    #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    newRectangles = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))    #try to detect face
    
    for knownface in faceList:
        bestDistance = 1000000
        chosenFace = (0,0,0,0)
        
        for rectangle in newRectangles:                                         #distance from each old face to new faces
            x1, y1, _, _ = knownface.boundingBox
            x2, y2, _, _ = rectangle
            distance = math.sqrt(pow(x2-x1,2) + pow(y2-y1,2))
            
            if distance < bestDistance:
                bestDistance = distance
                chosenFace = rectangle
        if bestDistance > maxDistance:
            pass
        else:
            dataset.append((chosenFace[0] - knownface.boundingBox[0], chosenFace[1] - knownface.boundingBox[1]))
            knownface.boundingBox = chosenFace
        
        x, y, w, h = knownface.boundingBox
        cv2.rectangle(frame, (x, y), (x+w, y+h), knownface.color, 2)

    if anterior != len(faceList):                                               #logfile
        anterior = len(faceList)
        log.info("faces: "+str(len(faceList))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    print("FPS: {:0.1f} imageStream size: {}".format(1.0/(time.time() - start_time),imageStream.qsize()))

pr.disable()                                                                    #end profiler
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())                                                             #print profiler stats

dx, dy = zip(*dataset)                                                          #print dx and dy to determine minimum length necessary for tracking
dx = [abs(number) for number in dx]
dy = [abs(number) for number in dy]

print("dx: ")
print(dx, max(dx))
print("dy: ")
print(dy, max(dy))

# When everything is done, release the capture
cv2.destroyAllWindows()
