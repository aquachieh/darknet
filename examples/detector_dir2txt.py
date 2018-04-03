'''
detector_dir2txt.py
'''

import sys, os
DN_PATH = "/.../darknet/"
sys.path.append(os.path.join(DN_PATH,"python/"))  #---add


#import from /.../darknet/python/darknet.py
#import darknet as dn
import darknet_s as dn  # print detect time
import pdb
import time 

start = time.time()

dn.set_gpu(2)   #0
net = dn.load_net(DN_PATH+"cfg/yolov3.cfg", DN_PATH+"yolov3.weights", 0)
meta = dn.load_meta(DN_PATH+"cfg/coco_person.data")    #coco.data


INPUT_IMG_PATH = DN_PATH + "bm1/"
text_file = open("/.../darknet/OUTPUT/bm1_c9.txt","w")
count = 1
t50 = 0
for fileName in os.listdir(INPUT_IMG_PATH):
    t2 = time.time() 
    r = dn.detect(net, meta, INPUT_IMG_PATH + fileName,0.9)    #---(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45)
    #r = dn.detect(net, meta, INPUT_IMG_PATH + fileName)
    t_1img = time.time()-t2
    print len(r)
    #print r[0][1],"---",r[0][2]
    for rr in r:
        TEXT_BBOX = fileName.split(".")[-2]+",person,"+str(int(rr[2][0]))+","+str(int(rr[2][1]))+","+str(int(rr[2][0]+rr[2][2]))+","+str(int(rr[2][1]+rr[2][3]))+"\n"
        #print rr[1]
        text_file.write(TEXT_BBOX)
    print "1img test time :{}".format(round(t_1img,3))
    count = count +1
    if 50<=count<100:
        t50 = t50 + t_1img

end = time.time() - start
print "Total Time Used: {}".format(end)
print "50 img test time :{}".format(round(t50,3))
print "50 img AVG test time :{}".format(round(t50/50,3))

text_file.close()
