import math
import torch

DTYPE = torch.float32

@torch.jit.script
def xdot(v, yaw):
    return v * torch.cos(yaw)

@torch.jit.script
def ydot(v, yaw):
    return v * torch.sin(yaw)

@torch.jit.script
def yawdot(v, delta, L):
    return v / L * torch.tan(delta)


def vdot(a):
    return 0   # Assume Acceleration is zero. i.e. Constant Velocity


class State:
    """
    State [Global_x, Global_y, Yaw(heading), Velocity]
        x: Global x         [m]
        y: Global y         [m]
        yaw: Heading angle  [rad]
        v: Velocity         [m/s]
    """
    def __init__(self, x=0., y=0., v=20., yaw=-math.pi/2):
        self.x = x
        self.y = y
        self.v = v
        self.yaw = yaw
        
    def __str__(self):
        return f'{self.x}, {self.y}, {self.yaw}, {self.v}'


class VehicleControl(object):
    def __init__(self, x=0., y=0., velocity=0., yaw=0., wheelbase=2.65):
        self.L = torch.tensor(wheelbase, dtype=DTYPE)
        self.state = State(x=torch.tensor(x, dtype=DTYPE), 
                                  y=torch.tensor(y, dtype=DTYPE), 
                                  v=torch.tensor(velocity, dtype=DTYPE), 
                                  yaw=torch.tensor(yaw, dtype=DTYPE), )

    def get_state(self, steer, duration):
        h = duration
        x1 = xdot(self.state.v, self.state.yaw)
        y1 = ydot(self.state.v, self.state.yaw)
        yaw1 = yawdot(self.state.v, steer, self.L)
        v1 = vdot(self.state.v)

        x2 = xdot(self.state.v + 0.5*h*v1, self.state.yaw + 0.5*h*yaw1)
        y2 = ydot(self.state.v + 0.5*h*v1, self.state.yaw + 0.5*h*yaw1)
        yaw2 = yawdot(self.state.v + 0.5*h*v1, steer, self.L)
        v2 = vdot(self.state.v + 0.5*h*v1)

        x3 = xdot(self.state.v + 0.5*h*v2, self.state.yaw + 0.5*h*yaw2)
        y3 = ydot(self.state.v + 0.5*h*v2, self.state.yaw + 0.5*h*yaw2)
        yaw3 = yawdot(self.state.v + 0.5*h*v2, steer, self.L)
        v3 = vdot(self.state.v + 0.5*h*v2)

        x4 = xdot(self.state.v + h*v3, self.state.yaw + h*yaw3)
        y4 = ydot(self.state.v + h*v3, self.state.yaw + h*yaw3)
        yaw4 = yawdot(self.state.v + h*v3, steer, self.L)
        v4 = vdot(self.state.v + h*v3)

        dx = h*(x1 + 2*x2 + 2*x3 + x4) / 6
        dy = h*(y1 + 2*y2 + 2*y3 + y4) / 6
        dyaw = h*(yaw1 + 2*yaw2 + 2*yaw3 + yaw4) / 6
        dv = h*(v1 + 2*v2 + 2*v3 + v4) / 6

        self.state.x += dx
        self.state.y += dy
        self.state.yaw += dyaw
        self.state.v += dv
        return self.state

    def update_velocity(self, v):
        self.state.v = torch.tensor(v, dtype=DTYPE)


