import logging
import os
import time

import gdown
import numpy

from deepface.commons import functions
import cv2
import onnx
import onnxruntime as ort
import numpy as np
import pandas as pd
# -------------------------------------------

def loadModel():
    return HSEmotionRecognizer()
class HSEmotionRecognizer:
    #supported values of model_name: enet_b0_8_best_vgaf, enet_b0_8_best_afew, enet_b2_8, enet_b0_8_va_mtl, enet_b2_7
    def __init__(self, model_name='enet_b0_8_best_vgaf'):
        home = functions.get_deepface_home()
        path = home + f"/.deepface/weights/{model_name}.onnx"
        if os.path.isfile(path) != True:
            print(f"{model_name}.onnx will be downloaded...")
            url='https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/onnx/'+model_name+'.onnx?raw=true'
            output = path
            gdown.download(url, output, quiet=False)
        self.is_mtl='_mtl' in model_name
        if '_7' in model_name:
            self.idx_to_class={0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'neutral', 5: 'sadness', 6: 'surprise'}
        else:
            self.idx_to_class={0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happiness', 5: 'neutral', 6: 'sadness', 7: 'surprise'}
        self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
        self.img_size=224 if '_b0_' in model_name else 260
        print("ORT device:", ort.get_device(), "OpenCV:", cv2.cuda.getCudaEnabledDeviceCount())
        if ort.get_device() == "GPU":
            # sessOptions = ort.SessionOptions()
            # sessOptions.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            self.ort_session = ort.InferenceSession(path,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        else:
            self.ort_session = ort.InferenceSession(path,providers=['CPUExecutionProvider'])

    def preprocess(self, img):
        x=cv2.resize(img,(self.img_size,self.img_size))/255
        x[..., 0] = (x[..., 0]-0.485)/0.229
        x[..., 1] = (x[..., 1]-0.456)/0.224
        x[..., 2] = (x[..., 2]-0.406)/0.225

        return x.transpose(2, 0, 1).astype("float32")[np.newaxis,...]

    def predict(self, face_img, verbose = 0):

        _,emotions = self.predict_emotions(face_img)

        return numpy.array([emotions])

    def predict_on_batch(self, face_img):
        return self.predict_multi_emotions(face_img)

    def predict_emotions(self,face_img, logits=True):
        scores=self.ort_session.run(None,{"input": self.preprocess(face_img)})[0][0]
        if self.is_mtl:
            x=scores[:-2]
        else:
            x=scores
        pred=np.argmax(x)
        if not logits:
            e_x = np.exp(x - np.max(x)[np.newaxis])
            e_x = e_x / e_x.sum()[None]
            if self.is_mtl:
                scores[:-2]=e_x
            else:
                scores=e_x
        return self.idx_to_class[pred],scores

    def predict_multi_emotions(self, face_img_list, logits=True):
        imgs = [self.preprocess(face_img) for face_img in face_img_list]
        imgs = np.concatenate(imgs, axis=0)
        scores=self.ort_session.run(None, {"input": imgs})[0]
        if self.is_mtl:
            preds=np.argmax(scores[:,:-2], axis=1)
        else:
            preds=np.argmax(scores, axis=1)
        if self.is_mtl:
            x=scores[:,:-2]
        else:
            x=scores
        pred=np.argmax(x[0])
        if not logits:
            e_x = np.exp(x - np.max(x, axis=1)[:,np.newaxis])
            e_x = e_x / e_x.sum(axis=1)[:,None]
            if self.is_mtl:
                scores[:, :-2]=e_x
            else:
                scores=e_x

        return [self.idx_to_class[pred] for pred in preds], scores
