import string,serial,cv2
import tkinter as tk
import numpy as np

from PIL import Image
from PIL import ImageTk

# window = tk.Tk()
# window.title("Join")
# window.geometry("600x600")
# window.configure(background='grey')

# panel = tk.Label(window)

def captureImage():
    ser = serial.Serial('COM5', 256000, 8, 'N', 1, timeout=1)
    ser.flushInput()
    
    while(True):
        #start_time = time.time()
        line = b' '
        jpeg_start=b'\xff\xd8'
        jpeg_end=b'\xff\xd9'
        message = None
        
        line = ser.read(128)
        index = line.find(jpeg_start)
        if index != -1:                                     #if message start bytes were found
            substrings = line.split(jpeg_start)
            message = jpeg_start + substrings[-1]
            while True:
                line = ser.read(2048)
                index_end = line.find(jpeg_end)
                if index_end != -1:                         #if we found the end of the image
                    substrings = line.split(jpeg_start)
                    message = message + substrings[0] + jpeg_end
                    break
                else:
                    message += line

        if message != None:
            frame = cv2.imdecode(np.fromstring(message, dtype=np.int8), 1)  #decode into a numpy array
            
            return frame
            
            # #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
            # img = ImageTk.PhotoImage(data=message)

            # #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
            # panel.configure(image=img)
            # #The Pack geometry manager packs widgets in rows or columns.
            # panel.pack(side = "bottom", fill = "both", expand = "yes")
            
            print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
            # window.update()
