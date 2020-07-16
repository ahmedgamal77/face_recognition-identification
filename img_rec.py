import face_recognition as fr
import cv2
import numpy as np
import os,shutil
from PIL import Image
import pickle
from function import *
 





   

def get_encoded_faces():
    

    with open('face_encodes.pickle', 'rb') as handle:
        b = pickle.load(handle)
    
    return b
encoded_faces=get_encoded_faces()
def classify_face(img,flag):
    
    known_faces_encodes=list(encoded_faces.values())#get list of known faces encodes
    knwon_Faces_names =list(encoded_faces.keys())#get the names of known faces
    image=cv2.imread(img)#load the unkown image
    image1=fr.load_image_file(img)
    unkown_face_locations=fr.api.face_locations(image1)#locate the positions of faces in image
    unkown_face_encode=fr.api.face_encodings(image1,unkown_face_locations)#encode each face
    unkown_face_names=list() #list for the names of the unkown faces in the image
    for face_enc in unkown_face_encode: #get the encode of every face in the image
        name="unkown"
        matches=fr.compare_faces(known_faces_encodes,face_enc) # search for the unkown face in the known faces to get its name if its known 
        face_distances = fr.face_distance(known_faces_encodes,face_enc)
        best_match = np.argmin(face_distances)
        if matches[best_match]:
            name=knwon_Faces_names[best_match]
        unkown_face_names.append(name)
    for(top,right,bottom,left), name in zip(unkown_face_locations,unkown_face_names):
        cv2.rectangle(image,(left-20,top-20),(right+20,bottom+20),(255,0,0),2)
        cv2.rectangle(image,(left-21,bottom+22),(right+21,bottom+55),(255,0,0),-2)
        cv2.putText(image,name,(left-15,bottom+40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    if(flag==1):
        filename=img.split(".")[0]+"after.jpg"
        cv2.imwrite(filename,image)
        shutil.copy(filename , ".\_after")
        os.remove(filename)
    
    return unkown_face_names



def learning(img):
    
    image=fr.load_image_file(img)
    unkown_face_locations=fr.api.face_locations(image)
    image1=Image.open(img)
    flag=0
    
    for loc in unkown_face_locations:
        im=image1.crop((loc[3] ,loc[0] ,loc[1] ,loc[2]))
        im.save("face.jpg")
        faces=classify_face("face.jpg",0)
        if(len(faces)==0):
            continue
        if(faces[0]=="unkown"):
            dec=input("there is an unkown person do you want to add it to the database? [y]/[n]")
            if(dec=='y'):
                unkow_face=cv2.imread("face.jpg")
                #cv2.imshow('ImageWindow', unkow_face)
                #cv2.waitKey(0)
                im.show()
                name=input("who is this ?")
                cv2.imwrite(name+".jpg",unkow_face)
                shutil.copy(name+".jpg",".\database")
                os.remove(name+".jpg")
                encoded_faces.update(encode_database("database/"+name+".jpg"))
                with open('face_encodes.pickle', 'wb') as handle:
                    pickle.dump(encoded_faces, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
                
    os.remove("face.jpg")
    
    
       
    names=classify_face(img,1)
    return names
    



  

#print(classify_face("face.jpg",1))
learning("kk.jpg")
