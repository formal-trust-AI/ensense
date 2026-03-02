import utils

from options import *
import pandas as pd
import math

def segment_idx( v, guards ): # TODO : should be a binary search
    val = 0
    for ival, vi in enumerate(guards):
        if v < vi: break
        val = ival+1
    return val

    # for i,vi in enumerate(vs):
    #     if v >= vi: break        
    # first = 0
    # last = len(vs)-1
    # found = False

    # while first<=last and not found:
    #     pos = 0
    #     midpoint = (first + last)//2
    #     if alist[midpoint] == item:
    #         pos = midpoint
    #         found = True
    #     else:
    #         if item < alist[midpoint]:
    #             last = midpoint-1
    #         else:
    #             first = midpoint+1
                
    # return (pos, found)

def interval_distance( p1, p2 ):
    if isinstance(p1, tuple) and not isinstance(p2, tuple):
        return interval_distance( p2, p1)
    if isinstance(p1, tuple) and isinstance(p2, tuple):
        if p1[1] < p2[0]: return interval_distance(p1[1],p2[0])
        if p2[1] < p1[0]: return interval_distance(p2[1],p1[0])
        return 0
    if isinstance(p2, tuple):
        if p1 < p2[0]: return interval_distance(p1,p2[0])
        if p2[1] < p1: return interval_distance(p2[1],p1)
        return 0
    else:
        return (p1-p2)**2
    
def interval_distanceL1( p1, p2 ):
    if isinstance(p1, tuple) and not isinstance(p2, tuple):
        return interval_distanceL1( p2, p1)
    if isinstance(p1, tuple) and isinstance(p2, tuple):
        if p1[1] < p2[0]: return interval_distanceL1(p1[1],p2[0])
        if p2[1] < p1[0]: return interval_distanceL1(p2[1],p1[0])
        return 0
    if isinstance(p2, tuple):
        if p1 < p2[0]: return interval_distanceL1(p1,p2[0])
        if p2[1] < p1: return interval_distanceL1(p2[1],p1)
        return 0
    else:
        return abs(p1 - p2)
    
def interval_distanceL1( p1, p2 ):
    if isinstance(p1, tuple) and not isinstance(p2, tuple):
        return interval_distanceL1( p2, p1)
    if isinstance(p1, tuple) and isinstance(p2, tuple):
        if p1[1] < p2[0]: return interval_distanceL1(p1[1],p2[0])
        if p2[1] < p1[0]: return interval_distanceL1(p2[1],p1[0])
        return 0
    if isinstance(p2, tuple):
        if p1 < p2[0]: return interval_distanceL1(p1,p2[0])
        if p2[1] < p1: return interval_distanceL1(p2[1],p1)
        return 0
    else:
        return abs(p1 - p2)
    

def interval_distanceL0( p1, p2 ):
    if isinstance(p1, tuple) and not isinstance(p2, tuple):
        return interval_distanceL0( p2, p1)
    if isinstance(p1, tuple) and isinstance(p2, tuple):
        if p1[1] < p2[0]: return interval_distanceL0(p1[1],p2[0])
        if p2[1] < p1[0]: return interval_distanceL0(p2[1],p1[0])
        return 0
    if isinstance(p2, tuple):
        if p1 < p2[0]: return interval_distanceL0(p1,p2[0])
        if p2[1] < p1: return interval_distanceL0(p2[1],p1)
        return 0
    else:
        return 0 if p1 == p2 else 1

def interval_distanceLinf(p1, p2):
    if isinstance(p1, tuple) and not isinstance(p2, tuple):
        return interval_distanceLinf(p2, p1)

    # interval vs interval
    if isinstance(p1, tuple) and isinstance(p2, tuple):
        a, b = p1
        c, d = p2
        return max(abs(a - c), abs(b - d))

    # point vs interval
    if isinstance(p2, tuple):
        x = p1
        c, d = p2
        return max(abs(x - c), abs(x - d))

    # point vs point
    return abs(p1 - p2)


def dist_segment(point1,point2, sensitive, segments):
    d = 0
    for i in range(len(point1)):
        if i in sensitive: continue
        if point1[i] == point2[i]: continue
        s1idx = segment_idx(point1[i],segments[i])
        s2idx = segment_idx(point2[i],segments[i])
        d += abs(s1idx-s2idx)
    return d

def dist(point1,point2, sensitive):
    d = 0
    for i in range(len(point1)):
        if i in sensitive: continue
        if point1[i] != point2[i]: d += interval_distanceL0(point1[i],point2[i])
    return d

def distL1(point1,point2, sensitive):
    d = 0
    # print(point1,point2,sensitive)
    for i in range(len(point1)):
        if i in sensitive: continue
        if point1[i] != point2[i]: d = d +  interval_distanceL1(point1[i],point2[i]) # (point1[i] - point2[i])**2 
    dis = math.sqrt(d)
    # print('L2 distance:', dis)
    return dis


def distLinf(point1,point2, sensitive):
    d = 0
    # print(point1,point2,sensitive)
    for i in range(len(point1)):
        if i in sensitive: continue
        if point1[i] != point2[i]: 
            distt =  interval_distanceLinf(point1[i],point2[i]) # (point1[i] - point2[i])**2 
            if distt > d:
                d = distt
    dis = math.sqrt(d)
    # print('L2 distance:', dis)
    return dis


def distL2(point1,point2, sensitive):
    d = 0
    # print(point1,point2,sensitive)
    for i in range(len(point1)):
        if i in sensitive: continue
        if point1[i] != point2[i]: d = d +  interval_distance(point1[i],point2[i]) # (point1[i] - point2[i])**2 
    dis = math.sqrt(d)
    # print('L2 distance:', dis)
    return dis


def data_distance(data, points, sensitive, segments = {}, dist_type = "L0"):    
    def find_min(dist_measure):
        min_d,min_point = float("inf"), None
        min_d2,min_point2 = float("inf"), None
        for index, row in data.iterrows():
            pointp = row.to_list()
            d = dist_measure(points[0], pointp, sensitive)
            if d < min_d: min_d, min_point = d, pointp
            if len(points) > 1:
                d2 = dist_measure(points[1], pointp, sensitive)
                if d2 < min_d2: min_d2, min_point2 = d2, pointp
        if len(points) > 1 and min_d2 < min_d:
            return (1,min_d2,min_point2)
        return (0,min_d,min_point)
    if len(points)==2 : sensitive = []
    if dist_type == "L0": (idx,min_d,min_point) = find_min(dist)
    if dist_type == 'L1':(idx,min_d,min_point) = find_min(distL1)
    if dist_type == "L2": (idx,min_d,min_point) = find_min(distL2)
    if dist_type == "Linf": (idx,min_d,min_point) = find_min(distLinf)
    if dist_type == "SegmentL1":
        if len(points)==2 : sensitive = []
        idx,min_d, min_point = 0, float("inf"), None
        idx2,min_d2, min_point2 = float("inf"), None
        for index, row in data.iterrows():
            pointp = row.to_list()
            d = dist_segment( points[0], pointp, sensitive, segments )
            if d < min_d: min_d, min_point = d,pointp
            if len(points) > 1:
                d2 = dist_segment( points[1], pointp, sensitive, segments )
                if d2 < min_d2: min_d2, min_point2 = d2,pointp
        if len(points) > 1 and min_d2 < min_d:
            idx,min_d, min_point = 1, min_d2, min_point2
    
    print('================== Data conformity analysis ===============')
    print(f'Distance from data distype {dist_type}: {min_d}')
    for i in range(len(min_point)):
        if i in sensitive:
            min_point[i] = "__"
            continue
        if points[idx][i] != min_point[i]:
            min_point[i] = (points[idx][i],min_point[i])
    print('Nearest point in data:', min_point)
    print('===========================================================')

def compute_data_distance( points, f, feature_names, n_features, trees, options,dist_type="L2"):
    if not isinstance(points[0], list): points = [points]
    assert(len(points[0]) == len(feature_names))
    if options.data_file:
        data = pd.read_csv(options.data_file,nrows=10000)
        feature_list = [""]*len(feature_names)
        for k in feature_names:
            feature_list[k] = feature_names[k]
        missing_cols = [col for col in feature_list if col not in data.columns]
        for col in missing_cols: 
            missing_indices = feature_list.index(col)  
            print(f"{col} is missing point {points[0][missing_indices]} in data, filling with {points[0][missing_indices]}")  
            data[col] = points[0][missing_indices]
        data = data[feature_list]
        # dist_type = "SegmentL1"
        # dist_type = "L0"
        if dist_type == "SegmentL1":
            segments = utils.feature_segments(trees,n_features)
            data_distance( data, points, f, segments=segments, dist_type="SegmentL1" )
        else:
            data_distance( data, points, f, dist_type=dist_type) 
    else:
        print("No data file is provided!")

