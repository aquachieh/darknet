

import sys, os
DN_PATH = "/home/sc/darknet/"
sys.path.append(os.path.join(DN_PATH,"python/"))  #---add


#import from /home/sc/darknet/python/darknet.py
#import darknet as dn
import darknet_sophie as dn
import pdb
import time 

start = time.time()

dn.set_gpu(2)   #0
net = dn.load_net(DN_PATH+"cfg/yolov3.cfg", DN_PATH+"yolov3.weights", 0)
meta = dn.load_meta(DN_PATH+"cfg/coco.data")


INPUT_IMG_PATH = DN_PATH + "sc_data/people_bm2/"
count = 1
t50 = 0
for fileName in os.listdir(INPUT_IMG_PATH):
    t2 = time.time() 
    r = dn.detect(net, meta, INPUT_IMG_PATH + fileName)
    #r = dn.detect(net, meta, INPUT_IMG_PATH + fileName)
    #---detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45)
    t_1img = time.time()-t2
    #print r,len(r),r[0][0]=="person"
    print r
    print "1img test time :{}".format(round(t_1img,3))
    count = count +1
    if 50<=count<100:
        t50 = t50 + t_1img

end = time.time() - start
print "Total Time Used: {}".format(end)
print "50 img test time :{}".format(round(t50,3))
print "50 img AVG test time :{}".format(round(t50/50,3))

