from function import *
import os,shutil
import time

face_encodes=dict()
for dirpath,dname,fname in os.walk(".\database"):
        for f in fname:
            if f.endswith(".jpg") or f.endswith(".png"):
                path="database/"+f
                face_encodes.update(encode_database(path))



with open('face_encodes.pickle', 'wb') as handle:
    pickle.dump(face_encodes, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
