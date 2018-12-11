import numpy as np
from math import sin
import cv2
import os
np.set_printoptions(threshold=100000)
np.set_printoptions(linewidth=800)

#constant
print("constant")
height = 99
width = 99
amplitude = 100
relative_amplitude = 0.5*10**5
lhaplus_steps = 100
steps = 1000
pi = 3.14159265358979
f = 2.1*10**12
T = 1/f
c = 3.0*10**8
wavelength = c/f
print("wavelength = {}".format(wavelength))
delta_d = wavelength/100
delta_t = delta_d/(5*c)
zeros_x = np.zeros(width, dtype="float64")
zeros_y = np.zeros(height, dtype="float64")
mu = np.zeros((height,width), dtype="float64")
eps = np.zeros((height,width), dtype="float64")
mu_img = np.zeros((height,width,3), dtype="uint32")
eps_img = np.zeros((height,width,3), dtype="uint32")
sigma_img = np.zeros((height,width,3), dtype="uint32")
mu_rgb = np.zeros((height,width), dtype="uint32")
eps_rgb = np.zeros((height,width), dtype="uint32")
sigma_rgb = np.zeros((height,width), dtype="uint32")
conductor = np.zeros((height,width), dtype="uint32")
non_conductor = np.zeros((height,width), dtype="uint32")
_left_conductor = np.zeros((height,width), dtype="uint32")
_right_conductor = np.zeros((height,width), dtype="uint32")
_down_conductor = np.zeros((height,width), dtype="uint32")
_up_conductor = np.zeros((height,width), dtype="uint32")
_lr_conductor = np.zeros((height,width), dtype="float64")
_du_conductor = np.zeros((height,width), dtype="float64")
_left_non_conductor = np.zeros((height,width), dtype="uint32")
_right_non_conductor = np.zeros((height,width), dtype="uint32")
_down_non_conductor = np.zeros((height,width), dtype="uint32")
_up_non_conductor = np.zeros((height,width), dtype="uint32")
_lr_non_conductor = np.zeros((height,width), dtype="float64")
_du_non_conductor = np.zeros((height,width), dtype="float64")
V = np.zeros((height,width,steps), dtype="float64")
ix = np.zeros((height,width,steps), dtype="float64")
iy = np.zeros((height,width,steps), dtype="float64")
nx = np.zeros((height,width), dtype="float64")
ny = np.zeros((height,width), dtype="float64")
Hi = np.zeros((height,width,steps), dtype="float64")
Hi_mask = np.zeros((height,width,steps), dtype="uint32")

#variables
print("variables")
Ex = np.zeros((height,width), dtype="float64")
Ey = np.zeros((height,width), dtype="float64")
Hz = np.zeros((height,width), dtype="float64")
Exbef = np.zeros((height,width), dtype="float64")
Eybef = np.zeros((height,width), dtype="float64")
Hzbef = np.zeros((height,width), dtype="float64")
var = np.zeros((height,width), dtype="float64")

#functions
print("functions")
def down(i):
    var[:,0:height-1] = i[:,1:height]
    var[:,height-1] = zeros_x.copy()
    return var.copy()
def up(i):
    var[:,1:height] = i[:,0:height-1]
    var[:,0] = zeros_x.copy()
    return var.copy()
def left(i):
    var[0:width-1,:] = i[1:width,:]
    var[width-1,:] = zeros_y.copy()
    return var.copy()
def right(i):
    var[1:width,:] = i[0:width-1,:]
    var[0,:] = zeros_y
    return var.copy()


#set mu(magnetic permeability) and eps(permittivity) and sigma(conductivity)
print("set mu,eps,sigma")
mu_img = cv2.imread("mu.bmp",cv2.IMREAD_COLOR).transpose(1,0,2).astype("uint32")
eps_img = cv2.imread("eps.bmp",cv2.IMREAD_COLOR).transpose(1,0,2).astype("uint32")
sigma_img = cv2.imread("sigma.bmp",cv2.IMREAD_COLOR).transpose(1,0,2).astype("uint32")
mu_rgb = mu_img[:,:,0]*mu_img[:,:,1]*mu_img[:,:,2]
eps_rgb = eps_img[:,:,0]*eps_img[:,:,1]*eps_img[:,:,2]
sigma_rgb = sigma_img[:,:,0]*sigma_img[:,:,1]*sigma_img[:,:,2]
mu = 1.25663*10**(-6)*(mu_rgb).astype("float64")/3000.0
eps = np.where(eps_rgb == 255**3, np.inf, 1) * 8.85418782*10**(-12)*(eps_rgb).astype("float64")/16.0
sigma = (sigma_rgb).astype("float64")/23548.0

files = os.listdir("./")
if "Hi.npy" in files:
    conductor = np.load("conductor.npy")
    ix = np.load("ix.npy")
    iy = np.load("iy.npy")
    nx = np.load("nx.npy")
    ny = np.load("ny.npy")
    Hi = np.load("Hi.npy")
    Hi_mask = np.load("Hi_mask.npy")
else:
    #lhaplus
    print("lhaplus")

    #compute initial condition
    print("initial condition")
    conductor = np.where(eps_rgb == 255**3, 1, 0)
    _left_conductor = left(conductor)
    _right_conductor = right(conductor)
    _down_conductor = down(conductor)
    _up_conductor = up(conductor)
    _lrdu_conductor = _left_conductor + _right_conductor + _down_conductor + _up_conductor
    _lr_conductor = (_left_conductor*_right_conductor*conductor*(-0.5)+1.0) #if both l and r are conductors, then divide by 2
    _du_conductor = (_down_conductor*_up_conductor*conductor*(-0.5)+1.0)
    # compute n
    nx_right_cond = -(conductor - _left_conductor)*(1-conductor).copy()
    nx_left_cond = -(_right_conductor - conductor)*(1-conductor).copy()
    ny_up_cond = -(conductor - _down_conductor)*(1-conductor).copy()
    ny_down_cond = -(_up_conductor - conductor)*(1-conductor).copy()
    nx = nx_right_cond + nx_left_cond
    ny = ny_up_cond + ny_down_cond

    def lhaplus_update(i):
        global V
        V[45, 49,i] = amplitude*np.sin(2*pi*(i+1)*delta_t/T, dtype = "float64")
        V[53, 49,i] = -amplitude*np.sin(2*pi*(i+1)*delta_t/T, dtype = "float64")
        V[:,:,i] = ((1/4)*(left(V[:,:,i])*_left_conductor + right(V[:,:,i])*_right_conductor + down(V[:,:,i])*_down_conductor + up(V[:,:,i])*_up_conductor + (4-_lrdu_conductor)*conductor)) * conductor
    
    for i in range(steps):
        print("initial condition {} steps".format(i))
        # compute V
        for j in range(lhaplus_steps):
            lhaplus_update(i)
        #compute i
        ix[:,:,i] = 1/(sigma*delta_d)*((V[:,:,i]-left(V[:,:,i]))*_left_conductor*conductor + (right(V[:,:,i])-V[:,:,i])*_right_conductor*conductor)*_lr_conductor
        iy[:,:,i] = 1/(sigma*delta_d)*((V[:,:,i]-down(V[:,:,i]))*_down_conductor*conductor + (up(V[:,:,i])-V[:,:,i])*_up_conductor*conductor)*_du_conductor
        #compute Hi
        Hi[:,:,i] = (up(ix[:,:,i])*ny_down_cond + down(ix[:,:,i])*ny_up_cond - right(iy[:,:,i])*nx_left_cond - left(iy[:,:,i])*nx_right_cond).astype("float64")
        Hi_mask[:,:,i] = np.where(Hi[:,:,i] != 0,0,1)

    np.save("conductor",conductor)
    np.save("ix",ix)
    np.save("iy",iy)
    np.save("nx",nx)
    np.save("ny",ny)
    np.save("Hi",Hi)
    np.save("Hi_mask",Hi_mask)


#variables
non_conductor = 1-conductor
_left_non_conductor = left(non_conductor)
_right_non_conductor = right(non_conductor)
_down_non_conductor = down(non_conductor)
_up_non_conductor = up(non_conductor)
_lr_non_conductor = (_left_non_conductor*_right_non_conductor*(non_conductor)*(-0.5)+1.0)
_du_non_conductor = (_down_non_conductor*_up_non_conductor*(non_conductor)*(-0.5)+1.0)

#update E
def update_Ex():
    global Ex, Exbef
    var = Ex
    Ex = (1/(1+delta_t*(sigma/eps)))*Exbef + (delta_t/delta_d)*((Hz-down(Hz))*_down_non_conductor*non_conductor+(up(Hz)-Hz)*_up_non_conductor*non_conductor)*_du_non_conductor/(eps+delta_t*sigma)
    Exbef = var
def update_Ey():
    global Ey, Eybef
    var = Ey
    Ey = (1/(1+delta_t*(sigma/eps)))*Eybef + (delta_t/delta_d)*((Hz-left(Hz))*_left_non_conductor*non_conductor+(right(Hz)-Hz)*_right_non_conductor*non_conductor)*_lr_non_conductor/(eps+delta_t*sigma)
    Eybef = var
#update H
def update_Hz():
    global Hz, Hzbef
    var = Hz
    Hz = (Hzbef - (delta_t/delta_d)*(((Ey-left(Ey))*_left_non_conductor*non_conductor+(right(Ey)-Ey)*_right_non_conductor*non_conductor)*_lr_non_conductor - ((Ex - down(Ex)) * _down_non_conductor * non_conductor + (up(Ex) - Ex) * _up_non_conductor * non_conductor) * _du_non_conductor)/mu)*(non_conductor)
    Hzbef = var

#main
def tileconcat(tile_images):
    return cv2.vconcat([cv2.hconcat(hori_images) for hori_images in tile_images])
ixmax,ixmin = np.max(ix),np.min(ix)
iymax,iymin = np.max(iy),np.min(iy)
Himax,Himin = np.max(Hi),np.min(Hi)
Vmax,Vmin = np.max(V),np.min(V)
print("main")
for i in range(steps):
    print("main {} steps...".format(i), end=' ')
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    Hz = Hz * Hi_mask[:,:,i] + relative_amplitude*amplitude*Hi[:,:,i]
    update_Ex()
    Ex = Ex * (1-np.abs(ny))
    print("updated Ex", end=' ')
    update_Ey()
    Ey = Ey * (1-np.abs(nx))
    print("updated Ey", end=' ')
    update_Hz()
    Hz = Hz * Hi_mask[:,:,i] + relative_amplitude*amplitude*Hi[:,:,i]
    print("updated Hz", end=' ')
    tile_plot = tileconcat([
        [(Hz+ 255/2).astype("uint8"), ((Hi[:,:,i]-Himin)*255/(Himax-Himin)).astype("uint8"), (np.abs(nx)*255).astype("uint8"),],
        [(Hz+ 255/2).astype("uint8"), ((Hi[:,:,i]-Himin)*255/(Himax-Himin)).astype("uint8"), (np.abs(ny)*255).astype("uint8"),],
        [(Hi_mask[:,:,i]*255).astype("uint8"), ((ix[:,:,i]-ixmin)*255/(ixmax-ixmin)).astype("uint8"), ((iy[:,:,i]-iymin)*255/(iymax-iymin)).astype("uint8")]
    ])
    cv2.imshow("result",tile_plot)
    if cv2.waitKey(10) == ord("q"):
        exit()