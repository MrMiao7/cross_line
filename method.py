#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : MrM7
# @Time : 2024/9/6 15:46
import numpy as np


def get_distance_and_direction(point, line):
    """
    参数：
        point：目标点坐标，list。
        line：线段的两个点坐标，list。
    返回结果：
        area_in：目标是否在线段垂直区域内。防止误统计没产生跨线段的物体。
            用点与线段两端点形成的向量，计算与线段夹角的cos角值，都为正，则在区域内，否则，在区域外。
            cos(a,b)=ab/|a||b|，固只需计算a、b向量的点积即可。
        area_extra_in：目标是否在额外补偿区域内。防止目标跨线前在垂直区域内，而跨线后，目标位置超越距离阈值时已出垂直区域的情况。
            计算目标与线段两端点的距离，取小的(更靠近某一端点)，若小于补偿距离阈值，则在额外补偿区域内。
            只适用于在跨线前在垂直区域内，跨线后在补偿区域内(与跨线前方向相反)的情况。
        distance：目标与线段的距离。跨线后，大于距离阈值时，才认为跨线，可有效缓解目标框抖动的情况。
            计算方法为b×sin(a,b)=a×b/|a|。
        direction：目标在线段(向量)的方向，左边或右边。方向改变则认为产生跨越。
            用向量夹角的sin值计算，sin(a,b)=a×b/|a||b|，sin值大于0，则为左边(图像坐标y轴向下为正，输出1)，小于0，则为右边。
            固只需计算a、b向量的叉乘即可。
    """
    p = np.array(point, dtype=int)
    p1 = np.array(line[0], dtype=int)
    p2 = np.array(line[1], dtype=int)
    vector_1 = p - p1
    vector_2 = p - p2
    vector_3 = p2 - p1
    area_in = (np.dot(vector_1, vector_3) >= 0 and np.dot(vector_2, -vector_3) >= 0)
    area_extra_in = False
    if area_in:
        vc = np.cross(vector_1, vector_3)
        distance = abs(vc) / np.linalg.norm(vector_3)
        direction = 1 if vc > 0 else 0
    else:
        extra_cross_threshold = 40.0
        p_p1 = np.linalg.norm(vector_1)
        p_p2 = np.linalg.norm(vector_2)
        extra_distance = min(p_p1, p_p2)
        if extra_distance < extra_cross_threshold:
            area_extra_in = True
            distance = extra_distance
            vc = np.cross(vector_1, vector_3)
            direction = 1 if vc > 0 else 0
        else:
            distance = 99999.99
            direction = -1
    return area_in, area_extra_in, distance, direction


if __name__ == "__main__":
    point = [100, 80]
    line = [[50, 50], [200, 200]]
    area_in, area_extra_in, distance, direction = get_distance_and_direction(point, line)
    print(area_in, area_extra_in, distance, direction)
