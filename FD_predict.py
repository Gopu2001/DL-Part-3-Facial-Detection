################################################################################

'''IMPORTS'''

################################################################################

import os, sys
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from mtcnn.mtcnn import MTCNN
sys.stderr = stderr

################################################################################

'''FUNCTIONS'''

################################################################################

def detect_faces(pix, faces, save=False, show=False, draw=True):
    ax = plt.gca()
    facedata = []
    for face in faces:
        x, y, w, h = face['box']
        facedata.append(pix[y:y+h,x:x+w])
        if(draw):
            ax.add_patch(Rectangle((x,y), w, h, fill=False, color='red'))
            for key, value in face['keypoints'].items():
                ax.add_patch(Circle(value, radius=2, color='red'))
    if(show or save):
        for i in range(len(facedata)):
            img = Image.fromarray(facedata[i], 'RGB')
            if(save):
                img.save(str(i) + '.png')
            if(show):
                img.show()

################################################################################

'''MAIN'''

################################################################################

if __name__ == '__main__':
    file = 'test1.JPG'
    pixels = plt.imread(file)
    plt.imshow(pixels)
    detect_faces(pixels, MTCNN().detect_faces(pixels), save=True, draw=True)
    plt.show()
else:
    detect_faces(pixels, MTCNN().detect_faces(pixels), save=True, draw=False)
