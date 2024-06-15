import webbrowser as wb
import folium
import branca.colormap
from collections import defaultdict
from folium.plugins import HeatMap, MiniMap, MarkerCluster
from folium import plugins
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 绘制热力图
def draw_heatmap(map, data):
    ################统计每个地点样本个数绘制热力图#####################
    data1 = data.groupby(["longitude", "latitude"]).agg("count")
    # 获取纬度
    coord = np.array(data1.index)
    count = np.array(data1.iloc[:,0])
    # data2中为纬度，经度，对应地点样本个数
    data2 = np.array([[coord[i][1], coord[i][0], count[i]] for i in range(data1.shape[0])])
    # data3 = pd.DataFrame(data2)
    # data3.to_excel('tableau.xlsx')
    ############用来设置颜色棒#####################
    steps = 20
    colormap = branca.colormap.linear.YlOrRd_09.scale(data2[:,2].min(), data2[:,2].max()).to_step(steps)
    colormap.add_to(map)  # add color bar at the top of the map
    colormap = branca.colormap.linear.YlOrRd_09.scale(0, 1).to_step(steps)
    gradient_map = defaultdict(dict)
    for i in range(steps):
        gradient_map[1 / steps * i] = colormap.rgb_hex_str(1 / steps * i)
    HeatMap(data2, gradient=gradient_map).add_to(map)  # Add heat map to the previously created map

# 增加小地图
def draw_minimap(map):
    minimap = MiniMap(toggle_display=True,
                      tile_layer='Stamen Watercolor',
                      position='topleft',
                      width=200,
                      height=200)
    map.add_child(minimap)

def point_marker(data, map):
    coordinate = data[["longitude", "latitude"]]
    for name, row in coordinate.iterrows():
        folium.Circle(location=[row["latitude"], row["longitude"]], radius=15000, color='blue',fill=False).add_to(map)  # 标记颜色  图标

def cluster_marker(coordinate, map):
    marker_cluster = plugins.MarkerCluster().add_to(map)
    for name, row in coordinate.iterrows():
        folium.Marker(location=[row["latitude"], row["longitude"]]).add_to(marker_cluster)  # 标记颜色  图标
def map_make(data, coordinate):  #
    '''通过更换tiles得到不同的底图'''
    # tiles = 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7'  # 常规地图
    # tiles = 'https://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}'  # 卫星影像图
    # tiles = 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=8&ltype=11'  # 街道图
    tiles = 'https://webrd02.is.autonavi.com/appmaptile?lang=en&size=1&scale=1&style=8&x={x}&y={y}&z={z}' #纯英文对照地图
    # location为中心坐标点，格式为[纬度,经度]，zoom_start表示初始地图的缩放尺寸，数值越大放大程度越大
    map = folium.Map(location=[coordinate['latitude'].mean(), coordinate['longitude'].mean()], attr='Raw data sample distribution heat map', tiles=tiles, zoom_start=4.5, control_scale=True)
    # draw_minimap(map)  # 创建小地图
    draw_heatmap(map, data)  # 可以用来设置地图，如气候,GDP等等
    # cluster_marker(coordinate, map)  # 通过聚类的方式把点标记出来
    # point_marker(data, map)  #  直接在地图上把所有的点标记出来
    map.add_child(folium.LatLngPopup())  # 点击地图可以显示出相应经纬度信息
    map.save('./data/map.html')
    wb.open('./data/map.html')