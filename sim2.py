from nav_gym.obj.robot.robot import CarRobot
from nav_gym.obj.robot.robot_param import CarParam
from nav_gym.sim.config import Config
from nav_gym.map.util import load_map
from nav_gym.obj.geometry.util import rot,line_line, line_polygon
from nav_gym.obj.geometry.objects import Polygon
from nav_gym.sim.plot import plot_cars
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
n_cars = 5
cars = []
polygons = []
center_x = 25.
center_y = 25.
r = 15.
plot = False
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

if plot == True:
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
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


    if plot==True:
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

    end_time2 = time.time()
    print("*********** time cost with plot : ", end_time2-start_time)
    # plt.show()