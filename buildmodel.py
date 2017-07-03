# from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math


# Given intial_point generate same particles
def initilize_motion_model(x_co,y_co,number_of_particles=100):
    motion_model=np.zeros((number_of_particles,2))
    for pair in motion_model:
        pair[0]=y_co
        pair[1]=x_co
    return motion_model

def calculate_radius_average_prob(y_co,x_co,rk,radius,x_length, y_length):
    average_value = 0
    x_start =int(min(max(x_co-radius,0),x_length-1))
    x_end = int(min(max(x_co+radius,0),x_length-1))
    y_start = int(min(max(y_co-radius,0),y_length-1))
    y_end = int(min(max(y_co+radius,0),y_length-1))
    # print('PARTICL SCOPE:',x_start,x_end,y_start,y_end)
    average_value = rk[x_start:x_end,y_start:y_end].mean()
    return average_value


def calculate_particle_color_prob(motion_model, radius, rk, x_length, y_length):
    n=motion_model.shape[0]
    probablity_vector = np.zeros(n)
    # y x s
    for i in range(motion_model.shape[0]):
        probablity_vector[i]=calculate_radius_average_prob(motion_model[i,0],motion_model[i,1],rk,radius,x_length,y_length)
    sum_of_all = sum(probablity_vector)
    if sum_of_all!=0:
        probablity_vector = np.divide(probablity_vector,sum_of_all)
    return probablity_vector


def residual_resample(weights):
    N = len(weights)
    indexes = np.zeros(N, 'i')
    # take int(N*w) copies of each weight, which ensures particles with the
    # same weight are drawn uniformly
    num_copies = (np.floor(N*np.asarray(weights))).astype(int)
    print('num_copies:',len(num_copies))
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]): # make n copies
            indexes[k] = i
            k += 1
    # for example if 3/N, then we will have 3 copy!

    # use multinormal resample on the residual to fill up the rest. This
    # maximizes the variance of the samples
    residual = weights - num_copies     # get fractional part
    residual /= sum(residual)           # normalize
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1. # avoid round-off errors: ensures sum is exactly one
    # Make sure that the size is still N.
    indexes[k:N] = np.searchsorted(cumulative_sum, np.random.random(N-k))
    # print(indexes.shape)

    return indexes

#  RESIDUAL RESAMPLING
def resample_motion_model(motion_model,likelihood_vector,residual=True):
    n = len(likelihood_vector)
    if residual:
        # Residual resameple
        indexes=residual_resample(likelihood_vector)
    else:
        # Simple resampel
        indexes=np.random.choice(n, size=n, p=likelihood_vector)
    # print(indexes)
    print(resample_motion_model)
    print(likelihood_vector)
    print(indexes)
    # print(motion_model[indexes,:])
    new_model=motion_model[indexes,:]
    return new_model

   
# same shape of np_array
def build_motion_model(motion_model, x_length, y_length, v_mean=None, v_variance=None, variance= 15):
    number_of_particles=motion_model.shape[0]
    if v_mean is None or v_variance is None:
        new_motion_model=motion_model+variance*np.random.randn(number_of_particles,2)
    # Check range here
    new_motion_model[:,0]=np.floor(np.clip(new_motion_model[:,0],0,y_length-1))
    new_motion_model[:,1]=np.floor(np.clip(new_motion_model[:,1],0,x_length-1))
    new_motion_model=new_motion_model.astype(int)
    # print(new_motion_model)
    return new_motion_model



def build_color_model(firstimagecrop):
    B = 8 
    width = 256/B
    # print ('Patch shape',firstimagecrop.shape)
    # // : Floor division - division that results into whole number adjusted to the left in the number line
    color_bin = np.int_(np.floor(np.divide(firstimagecrop,width)))


    his = np.ones((8,8,8))
    (a,b) = firstimagecrop.shape[:-1]
    for i in range(0,a):
        for j in range(0,b):
            his[color_bin[i][j][0],color_bin[i][j][1],color_bin[i][j][2]] += 1
    return his


def build_color_probabilitymap(his, image):
    (m,n) = image.shape[:-1]
    B = 8 
    width = 256/B
    color_bin = np.int_(np.floor(np.divide(image,width)))
    rk = np.ones((m,n))


    for i in range(0,m):
        for j in range(0,n):
            num = his[color_bin[i][j][0],color_bin[i][j][1],color_bin[i][j][2]]
            # num = his[color_bin[i,j]]
            # print(his.shape,his[[2,2,3]])
            # num=1 rk=0 
            # num>1 1>rk>0
            # num --> +inf rk--> 1  rk increase
            rk[i][j] = 1-min(1/num,1)
    # 0->Black
    # 1->White
    # rk_log = np.log(rk)
    # print(rk)

    # plt.imshow(np.power(rk,1.0/10),cmap='gray')
    # plt.show()
    # print(rk_log)
    # mini = np.min(rk_log)
    # maxi = np.max(rk_log)
    # plt.imshow(rk_log,vmin=0, vmax=1,cmap='gray')
    
    return rk



def calculate_distance(center1,center2):
    print('Center here:',center1,center2)
    return math.sqrt((center1[0]-center2[0])**2+(center1[1]-center2[1])**2)

def calculate_distance_pair_center(pair,center):
    return math.sqrt((pair[1]-center[0])**2+(pair[1]-center[0])**2)

def calculate_mask_probability(particle_motion_model,initial_radius,center,variance_this_turn):
    likelihood_vector = np.ones(particle_motion_model.shape[0])

    if center is None:
        return likelihood_vector
    i = 0 
    # print('Shape',likelihood_vector.shape)
    for pair in particle_motion_model:
        dis = calculate_distance_pair_center(pair,center)
        # print('dis',dis)
        if  dis > math.sqrt(variance_this_turn)*initial_radius :
            likelihood_vector[i]=math.exp(-(dis)/(2*2))
        elif dis > 1.5*initial_radius:
            likelihood_vector[i]=0.4
        elif dis > 1.2*initial_radius:
            likelihood_vector[i]=0.6

        i += 1


    return likelihood_vector

