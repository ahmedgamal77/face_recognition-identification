import face_recognition as fr
import cv2
import numpy as np
import os,shutil
from PIL import Image
import pickle

def encode_database(img):
    encoded=dict()
    face=fr.load_image_file(img)
    en=fr.api.face_encodings(face)[0]
    encoded[(img.split(".")[0]).split("/")[1]]=en
    return encoded

    
