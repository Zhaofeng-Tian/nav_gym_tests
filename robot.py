import numpy as np
from math import pi,cos,sin,tan
from nav_gym.obj.geometry.util import rot
from nav_gym.obj.geometry.lidar import *
from nav_gym.obj.robot.robot_param import CarParam
import os
from skimage import io
from skimage.draw import polygon

class CarRobot:
    def __init__(self,id, param,initial_state = np.array([0.,0.,0.,0.,0.,]), history_len = 5,dt=0.2):
        self.id = id
        self.value_base = param.value_base # base value of car robots
        self.dv = param.dv # delta value, the id value difference between two id-neighboring robots
        self.id_value = self.calc_id_value()
        self.world_reso = param.world_reso
        self.state = initial_state #[x, y, theta, v, phi]
        self.history = np.tile(initial_state,(history_len,1)).reshape((history_len,len(initial_state)))
        self.shape = param.shape
        self.geo_r = param.geo_r
        self.disk_r = param.disk_r
        self.disk_num = param.disk_num
        self.og_disk_centers = param.disk_centers
        self.disk_centers = (rot(self.state[2])@self.og_disk_centers.T).T + self.state[:2]
        self.og_vertices = param.vertices # original vertice coords w.r.t. car center at (0,0), and yaw at 0.
        # self.vertices = param.vertices # [rl,rr,fr,fl,lc,gc]
        self.vertices = (rot(self.state[2])@self.og_vertices.T).T + self.state[:2]
        self.v_limits = param.v_limits
        self.a_limits = param.a_limits
        self.dt = dt
        # Lidar param
        self.fan_range = param.fan_range
        self.min_range = param.min_range
        self.max_range = param.max_range
        self.ray_num = param.ray_num
        # Lidar observations
        self.angles = generate_angles(self.fan_range, self.fan_range, self.ray_num, reso = 0.098, is_num_based = True)
        self.og_end_points = generate_ends(self.angles,self.max_range)
        self.end_points = ends_tf(self.og_end_points, self.state[2],self.state[:2])
        # print(" angles: ", self.angles)
        # print("ends show here : ", self.og_end_points.shape)
        # print("state[2], theta: ", self.state[2])
        # print("x, y: ", self.state[:2])
        # print("new ends: ",ends_tf(self.og_end_points, self.state[2],self.state[:2]))
        self.ranges = None
        self.points = None
        # self.ranges, self.points = generate_range_points(start=(self.state[0],self.state[1]),
        #                                                  ends=ends_tf(self.end_points, self.state[2],self.state[:2]),
        #                                                  map=map, polygons=polygons, circles=circles, max_range=self.max_range)

    @property
    def A(self,):
        x, y, theta, v, phi = self.state
        l = self.shape[0] # wheel_base
        return np.array([cos(theta)*self.dt,sin(theta)*self.dt,self.dt*tan(phi)/l]) 
    
    def move_base(self, cmd):
        v = np.array([0.,0.]) # actually executed v, which is bounded by v and a limits
        if cmd[0] >= self.state[3]: # cmd > v: accelerate demand
            v[0] = min(self.state[3]+self.a_limits[0,0]*self.dt, self.v_limits[0,0], cmd[0])
        elif cmd[0] < self.state[3]: # deaccelerate demand
            v[0] = max(self.state[3]+ self.a_limits[1,0]*self.dt, self.v_limits[1,0], cmd[0])
        if cmd[1] >= self.state[4]: #  turn left demand
            v[1] = min(self.state[4]+self.a_limits[0,1]*self.dt, self.v_limits[0,1], cmd[1])
        elif cmd[1] < self.state[4]: # deaccelerate demand
            v[1] = max(self.state[4]+ self.a_limits[1,1]*self.dt, self.v_limits[1,1], cmd[1])
        return v

    
    def update(self, cmd ):
        # 1. velocity state update 
        self.state[3:] = self.move_base(cmd)
        # 2. x, y, theta pose state update
        self.state[:3] += self.A *self.state[3]
        # 3. vertices update
        self.vertices = (rot(self.state[2])@self.og_vertices.T).T + self.state[:2]
        # 4. disk centers update
        self.disk_centers = (rot(self.state[2])@self.og_disk_centers.T).T + self.state[:2]
        # 5. history 
        self.history = np.delete(self.history, 0 , axis=0)
        self.history = np.insert(self.history, len(self.history), self.state, axis=0)
        # 6. ends
        self.end_points = ends_tf(self.og_end_points, self.state[2],self.state[:2])
    
    def sensor_update(self, map, polygons, circles):
        """
        Sensor updates after state updates to sychronize agents in the environment
        """
        # 5. Lidar related update
        no_ego_map = map.copy()
        self.remove_body(no_ego_map)
        self.ranges, self.points = generate_range_points(start=(self.vertices[4,0],self.vertices[4,1]),
                                                         ends=self.end_points,
                                                         map=no_ego_map, polygons=polygons, circles=circles, max_range=self.max_range)
    def map_based_sensor_update(self, map):
        no_ego_map = map.copy()
        self.remove_body(no_ego_map)
        self.ranges, self.points = map_based_generate_range_points(start=(self.vertices[4,0],self.vertices[4,1]),
                                                         ends=self.end_points,
                                                         map = no_ego_map)
    def map_id_sensor_update(self,map):
        self.ranges, self.points = map_id_generate_range_points(start=(self.vertices[4,0],self.vertices[4,1]),
                                                         ends=self.end_points, map = map,
                                                         id_value = self.id_value, dv = self.dv)

    def id_fill_body(self,map):
        # dv = 0.0001 # to calc map id value
        r = np.round(self.vertices[:4][:,1]/self.world_reso)
        c = np.round(self.vertices[:4][:,0]/self.world_reso)
        rr, cc = polygon(r, c)
        map[rr,cc] = self.value_base+self.dv*self.id # id=0: 0.9900; id=1:0.9901 how to differentiate? if 0.99+id*dv-dv/2 < value <0.99+id*dv+dv/2

    def fill_body(self, map):
        r = self.vertices[:4][:,1]
        c = self.vertices[:4][:,0]
        rr, cc = polygon(r, c)
        map[rr,cc] = 1

    def remove_body(self, map):
        r = self.vertices[:4][:,1]
        c = self.vertices[:4][:,0]
        rr, cc = polygon(r, c)
        map[rr,cc] = 0
    
    def calc_id_value(self):
        return self.value_base+self.dv*self.id

    def get_scans(self):
        return self.ranges
    def get_states(self):
        return self.state

# image_path = os.path.join(os.getcwd(),'map/racetrack.png')
# # img = mpimg.imread(path)

# img = io.imread(image_path, as_gray=True)/255.0
# param = CarParam()
# robot = CarRobot(id = 0, param = param, initial_state=np.array([10.,10.,0.,0.,0.,]))
# # robot.get_scans()
# print(robot.history)