#%%
from numba import njit
import matplotlib.pyplot as plt

from skimage.transform import hough_line, hough_line_peaks
from matplotlib import cm

from PIL import Image as im

import numpy as np
import cv2

@njit
def get_rgb(img, color_list0, delta):
    color_list = np.zeros((color_list0.shape[0], 3))
    max_idx = int(len(img)*0.1)
    for ci in range(color_list0.shape[0]):
        c = color_list0[ci]
        temp_img = np.zeros(len(img))
        for i in range(len(img)):
            # temp_img[i] = np.linalg.norm(np.subtract(img[i], c)*1.0)
            temp_img[i] = np.sum(np.abs(np.subtract(img[i], c)))/1.5
        temp_idx = np.argsort(temp_img)
        img2 = img[temp_idx]
        for i in range(max_idx):
            i1,i2 = img2[i], img2[i+1]
            # if np.linalg.norm(np.subtract(i1, i2)*1.0)>delta:
            if np.sum(np.abs(np.subtract(i1, i2)))/1.5>delta:
                break
        color_list[ci] = np.sum(img2[:i], axis=0)/i
        
    return color_list

def get_line(p1,p2):
    (xs,ys),(xe,ye) = p1,p2
    line_len = int(np.ceil(max(abs(xs-xe)+1, abs(ys-ye)+1)))
    x_list = np.round(np.linspace(xs,xe,line_len, endpoint=True)-0.1).astype('int')
    y_list = np.round(np.linspace(ys,ye,line_len, endpoint=True)-0.1).astype('int')
    return [x_list, y_list]

def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

def get_line_angle(start, end):
    angle = np.degrees(np.arctan2(end[1]-start[1], end[0]-start[0]))
    angle -= 90
    if angle<0:
        angle+=360
    if angle>360:
        angle-=360
    return angle

def dis_C2AB(A, B, C):
    AC, AB = C-A, B-A
    return abs(np.dot(AC, AB)/np.linalg.norm(AB))

def meter_GorR(img, draw_flag=False):
    def get_line_angle(start, end):
        angle = np.degrees(np.arctan2(end[1]-start[1], end[0]-start[0]))
        angle -= 90
        if angle<0:
            angle+=360
        if angle>360:
            angle-=360
        return angle
    if draw_flag:
        plt.imshow(img)
        plt.show()
    if img.shape[2]>3:
        img = img[:,:,:3]
    if np.max(img)<=1:
        img*=255
        img = np.array(img, dtype='int')
    # img = img[:,:,::-1]
    # default green, red, black value
    # color_list0 = np.array([[12, 67, 44], [200,   0,  20], [45, 45, 45]])
    color_list0 = np.array([[ 0, 57, 17], [255, 143,  55], [38, 30, 42]])
    # update green, red value to fit current image
    color_list = get_rgb(np.reshape(img, (img.shape[0]*img.shape[1], 3)).astype('int64'), color_list0, delta=50)
    
    # generate green mask (green can only use smaller color_range, because in some image green and black is close)
    color_range = 15
    g_img = cv2.inRange(img, color_list[0]-color_range, color_list[0]+color_range)/255
    # generate red mask (red need larger color_range, in some image red can be such different)
    color_range = 40
    r_img = cv2.inRange(img, color_list[1]-color_range, color_list[1]+color_range)/255
    
    # do "OPEN" operation on green mask and red mask
    CK_size = int(min(img.shape[:1])*0.015)
    if CK_size>1:
        g_img = cv2.morphologyEx(g_img, \
                                 cv2.MORPH_OPEN, np.ones((CK_size,CK_size), np.uint8), iterations=1)
        r_img = cv2.morphologyEx(r_img, \
                                 cv2.MORPH_OPEN, np.ones((CK_size,CK_size), np.uint8), iterations=1)

    if draw_flag:
        plt.imshow(g_img+r_img)
        plt.title('greed+red area')
        plt.show()


    # filter outside area
        # find outline circle by greed mask and red mask
    temp = np.array(np.where(g_img+r_img)).T
    temp2 = sorted(temp, key=lambda x:(x[0],x[1]))
    max_h, min_h = temp2[-1], temp2[0]
    temp2 = sorted(temp, key=lambda x:(x[1],x[0]))
    max_v, min_v = temp2[-1], temp2[0]
    
    if max_h[0]-min_h[0]>max_v[1]-min_v[1]:
        p1, p2, p3 = max_h, min_h, max_v
    else:
        p1, p2, p3 = max_v, min_v, max_h
        
    (cx,cy), r = define_circle(p1, p2, p3)
        
        # crop image in outline circle
    img2 = img.copy()
    for x in range(0, img2.shape[0]):
        temp = r**2-(x-cx)**2
        if temp<0:
            img2[x, :] = [0,0,0]
            continue
        
        y1 = int(round(cy+np.sqrt(temp)))
        y2 = int(round(cy-np.sqrt(temp)))
        if y1>y2:
            y1,y2 = y2,y1
        y1 = max(y1, 0)
        y2 = min(y2, img2.shape[1])
        img2[x, :y1] = [0,0,0]
        img2[x, y2:] = [0,0,0]
    
    # update black value using cropped image to fit current image
    color_list = get_rgb(np.reshape(img2, (img2.shape[0]*img2.shape[1], 3)).astype('int64'), color_list0[2:], delta=40)
    
    # generate black mask (black can only use smaller color_range, because in some image green and black is close)
    color_range = 15
    n_img = cv2.inRange(img2, color_list[0]-color_range, color_list[0]+color_range)/255
    # do "OPEN" operation on green mask and red mask
    CK_size = int(min(img.shape[:1])*0.005)
    if CK_size>1:
        n_img = cv2.morphologyEx(n_img, \
                                 cv2.MORPH_OPEN, np.ones((CK_size,CK_size), np.uint8), iterations=1)
    
    if draw_flag:
        plt.imshow(g_img+r_img)
        plt.scatter([max_h[1], min_h[1], max_v[1], min_v[1]], [max_h[0], min_h[0], max_v[0], min_v[0]])
        plt.scatter([cy],[cx])
        plt.title('greed+red area and outline points')
        plt.show()

        plt.imshow(img)
        plt.scatter([max_h[1], min_h[1], max_v[1], min_v[1]], [max_h[0], min_h[0], max_v[0], min_v[0]])
        plt.scatter([cy],[cx])
        ax = plt.gca()
        ax.add_artist(plt.Circle((cy, cx), r, color='w', fill=False))
        plt.title('outline circle & original image')
        plt.show()
        
        plt.imshow(n_img)
        plt.scatter([max_h[1], min_h[1], max_v[1], min_v[1]], [max_h[0], min_h[0], max_v[0], min_v[0]])
        plt.scatter([cy],[cx])
        ax = plt.gca()
        ax.add_artist(plt.Circle((cy, cx), r, color='w', fill=False))
        plt.title('outline circle & black area')
        plt.show()
        
    if draw_flag:
        plt.figure(figsize = (12,4))
        plt.subplot(1,3,1), plt.imshow(g_img)
        plt.title('green area')
        plt.subplot(1,3,2), plt.imshow(r_img)
        plt.title('red area')
        plt.subplot(1,3,3), plt.imshow(n_img)
        plt.title('black area')
        plt.show()

    # find needle by Hough transform
    # it1, it2 = head, end point of the needle line
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(n_img, theta=tested_angles)
    
    if draw_flag:
        plt.figure()
        ax = plt.gca()
        temp = ['red', 'green', 'blue']
        for _, angle, dist, c in zip(*hough_line_peaks(h, theta, d, num_peaks=3), temp):
            print(_, angle, dist)
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            ax.axline((x0, y0), slope=np.tan(angle + np.pi/2), c=c)
        plt.imshow(n_img)
        plt.show()
    
        # plt.figure(figsize=(8,12))
        # plt.imshow(h)
        # plt.show()

    _, angle, dist = hough_line_peaks(h, theta, d, num_peaks=1)
    angle = angle[0]

    s0 = np.ravel(dist * np.array([np.cos(angle), np.sin(angle)]))[::-1]
    v = (np.tan(angle + np.pi/2), 1)
        
    if abs(v[0])>1:
        k = (img.shape[0]-1-s0[0])/v[0]
        it1 = [img.shape[0]-1, s0[1]+k*v[1]]
        k = (0-s0[0])/v[0]
        it2 = [0, s0[1]+k*v[1]]
    else:
        k = (img.shape[1]-1-s0[1])/v[1]
        it1 = [s0[0]+k*v[0], img.shape[1]-1]
        k = (0-s0[1])/v[1]
        it2 = [s0[0]+k*v[0], 0]

    # find needle pointing direction
        # find points fall on needle line
    a = set(zip(*get_line(it1, it2)))
    b = set(zip(*np.where(n_img)))
    temp = list(a&b)
    
        # find crossing point of needle line and outline circle
    v = np.subtract(it1,(cx,cy))
    it21 = np.add((cx,cy),(v/np.linalg.norm(v))*r)
    v = np.subtract(it2,(cx,cy))
    it22 = np.add((cx,cy),(v/np.linalg.norm(v))*r)
    
        # set one of it2 or it2 to centroid of "points fall on needle line"
    centroid = np.average(temp, axis=0)
    if np.linalg.norm(it21-centroid)>np.linalg.norm(it22-centroid):
        it1 = centroid
    else:
        it2 = centroid
    
    line_list = get_line(it1, it2)    
    
    if draw_flag:
        plt.imshow(n_img)
        ax = plt.gca()
        ax.add_artist(plt.Circle((cy, cx), r, color='w', fill=False))
        plt.scatter(*np.array(temp).T[::-1])
        plt.plot(*np.array((it1, it2)).T[::-1])
        plt.scatter([centroid[1]], [centroid[0]])
        plt.scatter(*np.array((it21, it22)).T[::-1])
        plt.axis([0,img.shape[1],img.shape[0],0])
        plt.title('finding pointing direction')
        plt.show()
    
    nd_img = np.zeros((img.shape[0],img.shape[1]), dtype='int')

    for x,y in zip(*line_list):
        if 0<x<img.shape[0] and 0<y<img.shape[1]:
            nd_img[x, y] = 1

    if draw_flag:
        plt.imshow(nd_img)
        # plt.axis([0,img.shape[1],0,img.shape[0]])
        plt.show()

    g_img2 = g_img+nd_img
    r_img2 = r_img+nd_img

    if draw_flag:
        plt.imshow(g_img2)
        # plt.axis([0,img.shape[1],0,img.shape[0]])
        plt.title('pointing line + green area')
        plt.show()
        plt.imshow(r_img2)
        # plt.axis([0,img.shape[1],0,img.shape[0]])
        plt.title('pointing line + red area')
        plt.show()

    g_len, r_len = len(np.ravel(np.where(g_img2==2))), len(np.ravel(np.where(r_img2==2)))


        
    # calc needle angle
    for x,y in zip(*line_list):
        temp = np.sqrt((x-cx)**2+(y-cy)**2)
        if abs(temp-r)<1:
            angle = get_line_angle((cx,cy), (x,y))
            break
            
    if draw_flag: #draw_flag:
        plt.figure()
        plt.imshow(img)
        plt.plot([y,cy],[x,cx], lw = 3, c = ['r','g'][g_len>r_len])
        plt.axis([0,img.shape[1],img.shape[0],0])
        # plt.axis([0,img.shape[1],0,img.shape[0]])
        plt.title('result')
        plt.show()
    
    # calc meter_percent
    gr_img = np.logical_or(g_img, r_img).astype('int')
    r2 = r*0.95
    for x in range(0, gr_img.shape[0]):
        temp = r2**2-(x-cx)**2
        if temp<0:
            continue

        y1 = int(round(cy+np.sqrt(temp)))
        y2 = int(round(cy-np.sqrt(temp)))
        
        y1 = max(y1, 0)
        y2 = min(y2, img2.shape[1])
        
        if gr_img[x, y1]==1:
            gr_img[x, y1] += 1
        if gr_img[x, y2]==1:
            gr_img[x, y2] += 1
            
    gr_x, gr_y = np.where(gr_img==2)
    split_idx = gr_y<gr_img.shape[0]/2
    
    L_idx = np.argmax(gr_x[split_idx])
    R_idx = np.argmax(gr_x[~split_idx])
    
    L_p = (gr_x[split_idx][L_idx], gr_y[split_idx][L_idx])
    R_p = (gr_x[~split_idx][R_idx], gr_y[~split_idx][R_idx])
    
    if draw_flag:
        plt.imshow(gr_img)
        plt.scatter([L_p[1], R_p[1]], [L_p[0], R_p[0]], c='red')
        plt.show()
    
    L_angle = get_line_angle((cx,cy), L_p)
    R_angle = get_line_angle((cx,cy), R_p)
    if 0>R_angle-360>=-90:
        R_angle-=360
        
    meter_percent = (angle-L_angle)/(R_angle-L_angle)
        
    print(f'0:{L_angle} 1:{R_angle}')
    print(f'needle angle:{angle} ({round(meter_percent*100,2)}%)')
    
    if (g_len>r_len) and (5>r_len):
        return True , round(meter_percent,4)
    elif (r_len>=g_len) and (5>g_len):
        return False , round(meter_percent,4)
    else:
        return None , 0.0000

if __name__ == '__main__':
    img_path = './meter/test.jpg'
    img = cv2.imread(img_path)
    img = img.astype('uint8')
    img = img[:,:,::-1]
    print(meter_GorR(img, True))
