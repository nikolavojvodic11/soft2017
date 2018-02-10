import cv2
import numpy as np
import os
import glob
import time
from sklearn.externals import joblib
from skimage.feature import hog


from skimage.measure import label
from scipy import ndimage

from skimage.measure import regionprops
from skimage.morphology import *

from sklearn.datasets import fetch_mldata
from skimage import color
import Find_Line as fl

#ucitavam dobijeni klasifajer
#clf = joblib.load("digits_cls.pkl")

if os.path.exists('out.txt')==True:
    files=glob.glob('*.txt')
    for filename in files:
        os.unlink(filename)
        print"Obrisan fajl!"
else:
    text1="RA64/2014 Nikola Vojvodic    "
    saveFile = open('out.txt', 'w')
    saveFile.write(text1)
    saveFile.close()
    print "Uspesno otvoren fajl!"


for video_Number in range(10):
    video_Name= "Snimci/video-"+str(video_Number)+".avi"

    cap = cv2.VideoCapture(video_Name)

    import math


    # preuzeto sa http://www.fundza.com/vectors/point2line/index.html
    def dot(v, w):
        x, y = v
        X, Y = w
        return x * X + y * Y


    def length(v):
        x, y = v
        return math.sqrt(x * x + y * y)


    def vector(b, e):
        x, y = b
        X, Y = e
        return (X - x, Y - y)


    def unit(v):
        x, y = v
        mag = length(v)
        return (x / mag, y / mag)


    def distance(p0, p1):
        return length(vector(p0, p1))


    def scale(v, sc):
        x, y = v
        return (x * sc, y * sc)


    def add(v, w):
        x, y = v
        X, Y = w
        return (x + X, y + Y)


    def pnt2line(pnt, start, end):
        line_vec = vector(start, end)
        pnt_vec = vector(start, pnt)
        line_len = length(line_vec)
        line_unitvec = unit(line_vec)
        pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
        t = dot(line_unitvec, pnt_vec_scaled)
        r = 1
        if t < 0.0:
            t = 0.0
            r = -1
        elif t > 1.0:
            t = 1.0
            r = -1
        nearest = scale(line_vec, t)
        dist = distance(nearest, pnt_vec)
        nearest = add(nearest, start)
        return (dist, (int(nearest[0]), int(nearest[1])), r)


    def pnt2line2(pnt, start, end):
        line_vec = vector(start, end)
        pnt_vec = vector(start, pnt)
        line_len = length(line_vec)
        line_unitvec = unit(line_vec)
        pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
        t = dot(line_unitvec, pnt_vec_scaled)
        r = 1
        if t < 0.0:
            t = 0.0
            r = -1
        elif t > 1.0:
            t = 1.0
            r = -1
        nearest = scale(line_vec, t)
        dist = distance(nearest, pnt_vec)
        nearest = add(nearest, start)
        return (dist, (int(nearest[0]), int(nearest[1])), r)

    #nalazim poziciju linije na snimku
    pos_x1,pos_y1,pos_x2,pos_y2=fl.Line(video_Name)
    line = [(pos_x1,pos_y1),(pos_x2,pos_y2)]
    lista_mnist=[]

    def LeftCorner(img_gray,min_x,min_y,max_x,max_y):
        try:
            img=label(img_gray)
            regions=regionprops(img)
            newImg=np.zeros((28,28))
            for region in regions:
                bbox=region.bbox
                if bbox[0]<min_x:
                    min_x=bbox[0]
                if bbox[1] < min_y:
                    min_y = bbox[1]
                if bbox[2] > max_x:
                    max_x = bbox[2]
                if bbox[3] > max_y:
                    max_y = bbox[3]

            height = max_x - min_x
            width = max_y - min_y

            newImg[0:height, 0:width] = newImg[0:height, 0:width] + img_gray[min_x:max_x, min_y:max_y]

            return newImg

        except ValueError:
         pass

    def Find_Number(img_BW):
        try:
            label_img = label(img_BW)
            regions = regionprops(label_img)

            newImg = "newImage"

            i = 0
            minx = 500
            miny = 500
            maxx = -1
            maxy = -1

            for region in regions:
                bbox = region.bbox
                if bbox[0] < minx:
                    minx = bbox[0]
                if bbox[1] < miny:
                    miny = bbox[1]
                if bbox[2] > maxx:
                    maxx = bbox[2]
                if bbox[3] > maxy:
                    maxy = bbox[3]

            width = maxx - minx
            height = maxy - miny
            newImgLeft = np.zeros((28, 28))

            newImgLeft[0:width, 0:height] = newImgLeft[0:width, 0:height] + img_BW[minx:maxx, miny:maxy]
            newImg = newImgLeft;
            return newImg
        except ValueError:
            pass

    mnist = fetch_mldata('MNIST original')
    new_mnist_set=[]


    def findClosest(list,elem):
        temp = list[0]
        for obj in list:
            if distance(obj['center'], elem['center']) < distance(elem['center'], temp['center']):
                temp = obj


        return temp

    def inRange(items, item):
        retVal = []
        for obj in items:
            mdist = distance(obj['center'], item['center'])

            if (mdist < 20):
                retVal.append(obj)


        return retVal

    Id_Number=-1
    def nextId():
        global Id_Number
        Id_Number += 1
        return Id_Number

    def draw_Points(img, pointArr):
        cv2.circle(img, (pointArr[0]), 4, (25, 25, 255), 1)
        cv2.circle(img, (pointArr[1]), 4, (25, 25, 255), 1)


    def draw_All_Points(img, linesAll):
        s = 20
        for i in range(len(linesAll)):
            for x1, y1, x2, y2 in linesAll[i]:
                cv2.circle(img, (line[0]), 4, (2, 2, s), 1)
                cv2.circle(img, (line[1]), 4, (2, 2, s), 1)


        s = s + 20


    passed_Number=0
    sum_Numbers=0

    def main():

        kernel = np.ones((2, 2), np.uint8)

        boundaries = [([220, 220, 220], [255, 255, 255])]
        t = 0
        elements = []
        times = []

    #punim mnist

        i = 0
        minx = 500
        miny = 500
        maxx = -1
        maxy = -1


        print "mnist data " + format(len(mnist.data))
        for i in range(len(mnist.data)):
            mnist_img = mnist.data[i].reshape(28, 28)
            mnist_temp = color.rgb2gray(mnist_img) / 255.0 >= 0.88
            mnist_img_gray = (mnist_temp).astype('uint8')
            new_mnist_img = LeftCorner(mnist_img_gray, minx, miny, maxx, maxy)
            lista_mnist.append(new_mnist_img)

        while cap.isOpened():

            (lower,upper)=boundaries[0]

            lower_Bound = np.array(lower, dtype="uint8")
            upper_Bound = np.array(upper, dtype="uint8")

            start_time=time.time()
            ret,frame=cap.read()
            if not ret:
                break

            mask=cv2.inRange(frame,lower_Bound,upper_Bound)

            frame0=1.0*mask
            frame01=1.0*mask
            draw_Points(frame,line)



            frame0=cv2.dilate(frame0,kernel)
            frame0=cv2.dilate(frame0,kernel)


        #pronalazi objekte koji su pronadjeni na slici i jedinstveno ih oznacava i sadrzani su
        #u objektu labeled, a broj pronadjenih objekata se nalazi u promjenljivij nr_objects

            labeld,nr_obejcts=ndimage.label(frame0)
            objects=ndimage.find_objects(labeld)

            for k in range(nr_obejcts):
                loc=objects[k]
                (dxc,dyc)=((loc[1].stop - loc[1].start),(loc[0].stop - loc[0].start))
                (xc, yc) = ((loc[1].stop + loc[1].start) / 2,(loc[0].stop + loc[0].start) / 2)

                if(dxc>11 or dyc>11):
                    elem={'center' : (xc,yc),'size' : (dxc,dyc),'t' : t}

                    lst=inRange(elements,elem)
                    nn=len(lst)
                    if nn==0:

                        x1 = xc - 14
                        x2 = xc + 14
                        y1 = yc - 14
                        y2 = yc + 14

                        elem['id']=nextId()
                        #elem['vrednost'] = Take_Number(frame01[y1:y2, x1:x2])
                        elem['t']=t
                        elem['pass']=False
                        img_BW = color.rgb2gray(frame01[y1:y2, x1:x2]) >= 0.88
                        img_BW = (img_BW).astype('uint8')

                        newImg=Find_Number(img_BW)

                        minDifference = 8888
                        number = -10
                        for i in range(70000):
                            difference = 0
                            mnist_img = lista_mnist[i]
                            diff = mnist_img != newImg
                            difference = np.sum(diff)
                            if difference < minDifference:
                                minDifference = difference
                                number = mnist.target[i]
                        elem['number'] = number;
                        elements.append(elem)

                    else:

                        el=findClosest(lst,elem)
                        el['center']=elem['center']
                        el['t']=t

                for el in elements:
                    tt=t-el['t']
                    if(tt<3):
                        dist,pnt,r=pnt2line2(el['center'],(pos_x1,pos_y1),(pos_x2,pos_y2))
                        if r>0:
                            if dist<10:
                                if el['pass']==False:
                                    el['pass']=True

                                    (x,y)=el['center']
                                    x1 = x - 14
                                    x2 = x + 14
                                    y1 = y - 14
                                    y2 = y + 14

                                    print "Presao broj: " + format(el['number'])

                                    global passed_Number
                                    global sum_Numbers

                                    #sum_Numbers.append(el['vrednost'])
                                    sum_Numbers+=(el['number'])

                                    passed_Number += 1

            elapsed_time=time.time()-start_time
            times.append(elapsed_time * 1000)
            cv2.putText(frame,'Suma: ' + str(sum_Numbers),(450,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)


            cv2.imshow('Frame',frame)
            if cv2.waitKey(10) & 0xFF==ord('q'):
                break
        print"Suma:"+format(sum_Numbers)

        cap.release()
        cv2.destroyAllWindows()

        text = "Suma:   " + video_Name + "=" + format(sum_Numbers)

        saveFile=open('out.txt','a')
        saveFile.write(text)
        saveFile.close()
        print"Uspesno upisano!"
        Vide_time=np.array(times)
        print"Mean: %.2f ms" % (np.mean(Vide_time))

    main()
