from nav_gym.obj.robot.robot import CarRobot
from nav_gym.obj.robot.robot_param import CarParam
from nav_gym.sim.config import Config
from nav_gym.map.util import load_map
from nav_gym.obj.geometry.util import rot,line_line, line_polygon
from nav_gym.obj.geometry.objects import Polygon
import numpy as np
from math import cos, sin, pi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import random
import time

"""
Static Polygon, Circles represent static obstacles, will be draw on the map.
Dynamic Polygons, Circles represent cars and round robots, the geometry of them will be maintained dynamically.
"""
world_x = 50.
word_y = 50.
car_param = CarParam("normal")
print("car param, ",car_param)
static_polygons =[]
static_circles = []
dynamic_polygons = []
dynamic_circles = []
config = Config()
n_cars = 50
cars = []
polygons = []
center_x = 25.
center_y = 25.
r = 15.
plot = True

# def plot_cars(ax, cars):
#     for car in cars:
#         fcolor = 'g'
#         if car.id == 0:
#             fcolor = 'r'
#         rect = patches.Rectangle((car.og_vertices[1,0], car.og_vertices[1,1]), car.shape[0]+car.shape[1]+car.shape[2], 2*car.shape[3], linewidth=1, edgecolor='black', facecolor=fcolor)
#         tf = transforms.Affine2D().rotate(car.state[2]).translate(*(car.state[0],car.state[1]))
#         rect.set_transform(tf + ax.transData)

#         # Add the patch to the axis
#         ax.add_patch(rect)
def plot(img,cars):
    plt.cla()
    ax.imshow(img, origin = 'lower',cmap='gray',extent=[0,50,0,50])  
    for car in cars:
        for end in car.points:
            start = car.vertices[4].copy()
            x = [start[0], end[0]]
            y = [start[1], end[1]]
            ax.plot(x,y,color='blue')
        # print(" Range observation: ", car.ranges)
    plot_cars(ax, cars)
    plt.draw()
    plt.pause(0.02)


def plot_cars(ax,cars):
    for car in cars:
        fcolor = 'g'
        if car.id == 0:
            fcolor = 'r'
        rect = patches.Polygon(car.vertices[:4],linewidth=1, edgecolor='black', facecolor=fcolor)


        # Add the patch to the axis
        ax.add_patch(rect)


n_circles = random.randint(config.n_circles[0], config.n_circles[1])
n_cubes = random.randint(config.n_cubes[0], config.n_cubes[1])
circles = []



for i in range(n_cars):
    cars.append(CarRobot(id = i,
                         param= car_param, 
                         initial_state = np.array([center_x +r*cos(i*2*pi/n_cars), center_y+r*sin(i*2*pi/n_cars), i*2*pi/n_cars+pi, 0., 0. ])))
    polygons.append(Polygon(cars[i].vertices[:4]))

# cars.append(CarRobot(id=1, param=car_param, initial_state=np.array([40., 24.,3.15, 0.0, 0.0])))
img = load_map("racetrack.png")
map = img.copy()
# for polygon in static_polygons:
#     fill_poly(polygon,map)
# for circle in static_circles:
#     fill_disk(circle, map)

# print("vertices are: ",cars[0].vertices[:4])
# Create a polygon instance.
p1 = Polygon(cars[0].vertices[:4])
# print("Polygon's edges are: ",p1.edges)




if plot == True:
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # ax.set_xlim(0, 50)
    # ax.set_ylim(0,50)
    # self.ax.legend(loc='upper right')
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    


"""
Big Loop to update states and observations.
"""
for i in range(30):
    start_time = time.time()
    polygons = []
    # *************** States Update Loop Starts ***************** #
    for car in cars:
           
        car.update(np.array([1.,0.2]))
        
        polygons.append(Polygon(car.vertices[:4]))

    # *************** States Update Loop Ends *********************

    # *************** Observation Update Loop Starts **************
    for car in cars:
        temp_polygons = polygons.copy()
        temp_polygons.pop(car.id)
        car.sensor_update(img,temp_polygons,circles)
    end_time1 = time.time()
    print("*********** time cost w/o plot : ", end_time1-start_time)

    if plot == True:
        plot(img, cars)



    end_time2 = time.time()
    print("*********** time cost with plot : ", end_time2-start_time)
    # plt.show()