import cv2
import numpy as np
import os
import math
import sys
import json
import time
import argparse
import tensorflow as tf

from mrcnn import utils
from mrcnn import visualize 
import mrcnn.model as modellib
from mrcnn.model import log
from tracker import Tracker

sys.path.append(os.path.join("coco/"))  # Path dataset coco (Common Objects in Context)
import coco

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="path to input image")
args = vars(parser.parse_args())


# Directorio de logs
MODEL_DIR = os.path.join("logs/")

# Ruta del archivo de pesos de la red rcnn
COCO_MODEL_PATH = os.path.join("coco/mask_rcnn_coco.h5")

# Si no existe la ruta descargar los pesos a partir de la Release de Coco
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
# Configuramos numero de GPUs e imagenes por GPU
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Skip detections with < 95% confidence
    DETECTION_MIN_CONFIDENCE = 0.95

config = InferenceConfig()

  #Cargamos modelo RCNN (constructor)
model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
#Volcamos los pesos en el modelo RCNN
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Clases contempladas en la dataset de coco
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
              'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
              'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
              'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
              'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
              'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
              'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
              'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
              'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']
def asig_color():
  #Asigna un color a cada clase
  colors = [tuple(255 * np.random.rand(3)) for _ in range(512)]
  return colors
colors= asig_color() # Lista de Colores

def detecta_personas(boxes, masks, ids, scores, class_detected):

  personas = []
  masks_personas = []
  scores_personas = []
  n_instances = len(boxes)

  if not n_instances:
    print('NO INSTANCES TO DISPLAY')
  else:
    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        label = class_names[ids[i]]
        score = scores[i]
        
        if (label in class_detected): # Solo personas

          personas.append(boxes[i])
          masks_personas.append(masks[i])
          scores_personas.append(scores[i])

  return personas, masks_personas, scores_personas
  
def centros(boxes):
  centers = []
  n_dim = len(boxes)
  for i in range(n_dim):
    (y1, x1, y2, x2) = boxes[i]
    x = int(round((x1 + x2)/2.0))
    y = int(round((y1 + y2)/2.0))
    centro = np.array([y,x])
    centers.append(centro)
  return np.array(centers)
  
# Variables previas
writer = None
tracker = Tracker(150, 30, 5)
skip_frame_count = 0

input = str(args["input"])
video_in = "videos/" + input
video_out = "output/" + input.replace(".mp4", "_out.avi")
class_detected = 'person' # Detectaremos solo personas

#Leemos video
capture = cv2.VideoCapture(video_in) # Leemos video
#Ajustamos la resolución de los frames a 720x480
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps = capture.get(cv2.CAP_PROP_FPS) # Obtenermos fps
print(fps)

try:
  prop = cv2.CAP_PROP_FRAME_COUNT
  total = int(capture.get(prop))
  print("[INFO] {} Frames totales: ".format(total))

# Exception por si no se pueden recuperar los frames
except:
  print("[INFO] No se detectó el número de Frames")
  total = -1
  
'''
Bucle de imágenes
'''

while True:

  ret, frame = capture.read() # Capturamos frame

  if ret == True: # En caso de capturarlo correctamente

    start = time.time() # Ponemos en marcha timer

    results = model.detect([frame], verbose=0) # Detectamos objetos
    r = results[0] # Solo una imagen

    personas, people_masks, people_scores = detecta_personas( r['rois'],
    r['masks'], r['class_ids'], r['scores'], class_detected)

    centers = centros(personas)

    if (len(centers) > 0):

      tracker.update(centers)

      for j in range(len(tracker.tracks)):

        try:
          c1, c2 = centers[j]
          y1, x1, y2, x2 = personas[j]
          #cv2.circle(frame,(c1,c2), 6, (0,0,0),-1)
        except:
          pass

        d1 = abs(y1-c1)
        d2 = abs(x1-c2)
        if (len(tracker.tracks[j].trace) > 1):

          x = int(tracker.tracks[j].trace[-1][0,0])
          y = int(tracker.tracks[j].trace[-1][0,1])
          tl = (x-d2,y-d1)
          br = (x+d2,y+d1)
          cv2.rectangle(frame,tl,br,colors[j],2)
          cv2.putText(frame,"Persona: " + str(tracker.tracks[j].trackId+1), (x1,y2),cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[j],2)
          cv2.circle(frame,(x,y), 1, colors[j],2)

      end = time.time() # Ponemos en marcha timer

      if writer is None:
            # Escribimos en el video de salida
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(video_out, fourcc, fps, (frame.shape[1], frame.shape[0]), True)
      if total > 0:
            elap = (end - start)
            print("[INFO] Tiempo que ha tardado el frame: {:.4f} s".format(elap))
            print("[INFO] Tiempo estimado: {:.4f}".format(elap * total))
   


      writer.write(frame) # Escribimos en el disco

  else:
    break

writer.release()
capture.release()
