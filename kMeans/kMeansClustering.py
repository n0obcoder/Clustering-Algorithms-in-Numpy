import numpy as np
import cv2
import os,sys,glob,pdb
from matplotlib import pyplot as plt

disp = True

###
#numpy.random.normal(loc=0.0, scale=1.0, size=None)
###
############################# DATA GENERATION #############################
def gen_data(data_points,loc):
    #print('data_points: ', data_points)
    #print('loc: ', loc)
    x_center = loc[0]
    y_center = loc[1]
    x = np.random.normal(size= data_points, loc= x_center)*100
    y = np.random.normal(size= data_points, loc= y_center)*100
    data= np.vstack((x,y))
    #print(x.shape)
    #print(y.shape)
    #print(data.shape)
    #k=0
    #print(x[k])
    #print(y[k])
    #print(data)
    #print(x.shape,y.shape)
    #exit()
    data = data.astype(np.int32)
    x,y = data
    #print('x.shape: ', x.shape)
    #print('y.shape', y.shape)
    #print(x)
    #pdb.set_trace()
    return x, y

results_dir= 'results'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

#...................................
x_data = np.array((),dtype=np.int32)
y_data = np.array((),dtype=np.int32)
#...................................
x, y = gen_data(100,[5,5]) #plt.hist(x) shows that is a normal distribution
plt.scatter(x, y)
x_data = np.hstack((x_data, x))
y_data = np.hstack((y_data, y))
#...................................
x, y = gen_data(50,[-2,4]) 
plt.scatter(x, y)
x_data = np.hstack((x_data, x))
y_data = np.hstack((y_data, y))
#...................................
x, y = gen_data(300,[2,3]) 
plt.scatter(x, y)
x_data = np.hstack((x_data, x))
y_data = np.hstack((y_data, y))
#...................................
#plt.scatter(x_data,y_data)
plt.title('k-Means Clustering')
plt.xlabel('Some Feature x')
plt.ylabel('Some Feature y')
plt.grid()
plt.savefig(results_dir + '/generated_data.jpg')
plt.show()
print('>>>>>>>>>>>>>>>>>>>>>>>>>> DATA POINTS GENERATED')
    
############################# k-Means #############################

#find out the extremes of our data space

x_min= np.min(x_data)
x_max= np.max(x_data)
y_min= np.min(y_data)
y_max= np.max(y_data)
print('x_min, x_max, y_min, y_max: ', x_min, x_max, y_min, y_max)

#set the value of 'k'
color_list = ['Blue', 'Green', 'Yellow', 'Pink', 'Purple']
k= 3 #you can change the value of k till 5 because the color_list that i have defined has only 5 colors. you can add more colors to the color_list and then change k to higher values.

#randomly initialize k number of center points
kcenters = []

for i in range(k):

    x_center= np.random.randint(x_min, x_max + 1)
    y_center= np.random.randint(y_min, y_max + 1)
    center = [x_center, y_center]
    kcenters.append(center)
    
    print('k center point {} initialized at ({}, {})'.format(i+1, x_center, y_center))

print('kcenters: ', kcenters)

def display_kcenters(kcenters):
    for i in range(len(kcenters)):
        plt.scatter(kcenters[i][0], kcenters[i][1], color='Red')
    #plt.show()
    
initial_data_point_color = 'Gray'    
display_kcenters(kcenters)
plt.scatter(x_data, y_data, color= initial_data_point_color)
plt.title('before kMeans Clustering')
plt.xlabel('Some Feature x')
plt.ylabel('Some Feature y')
plt.grid()
plt.savefig(results_dir + '/before_kmeans.jpg')
plt.show()

color_info_list = [initial_data_point_color]*len(x_data)

#print(color_info_list)
#print(len(color_info_list))

print('>>>>>>>>>>>>>>>>>>>>>>>>>> KCENTERS INITIALIZED')

def assign_cluster(x, y, kcenters):
    distances = []
    for i in range(len(kcenters)):
        #print(i)
        dist = np.sqrt(np.square(x - kcenters[i][0]) + np.square(y - kcenters[i][1]))
        #print(dist)
        distances.append(dist)
    distances = np.array(distances)
    #print(distances)
    #print(type(distances))
    
    return np.argmin(distances)

def get_mean_shit_value(old_point, new_point):
    #pdb.set_trace()
    mean_shift_value = np.sqrt(  np.square(old_point[0] - new_point[0]) + np.square(old_point[1] - new_point[1])  )
    return mean_shift_value


iterations = 100
stop=[False]*k

for n in range(iterations):
    #iterate over the data points and assign them to one of the k clusters
    print('iteration number {}'.format(n))
    
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        
        #print(type(x)) #<type 'numpy.int32'>
        cluster_index= assign_cluster(x, y, kcenters)
        
        color_info_list[i] = color_list[cluster_index]
        #print('point {} goes {}'.format(i+1, color_list[cluster_index]))
        

    plt.scatter(x_data, y_data, color= color_info_list)
    display_kcenters(kcenters)
    plt.title('k-Means iter {}'.format(n))
    plt.xlabel('Some Feature x')
    plt.ylabel('Some Feature y')
    plt.grid()
    plt.savefig(results_dir + '/iter{}.jpg'.format(n))
    plt.show()

    #calculate the new position of the centers

    for i in range(len(kcenters)):
        
        x_coords_list=[]
        y_coords_list= []
        for j in range(len(x_data)):
            if color_info_list[j] == color_list[i]:
                x_coords_list.append(x_data[j])
                y_coords_list.append(y_data[j])
        
        x_mean = np.mean(x_coords_list).astype(np.int32)
        y_mean = np.mean(y_coords_list).astype(np.int32)
        #print(type(x_mean))
        #print('kcenters[{}] shifted from {} to {}'.format(i, kcenters[i],[x_mean,y_mean]))
        
        mean_shift_value = get_mean_shit_value(kcenters[i], [x_mean, y_mean])
        if disp:
            print('mean_shift_value: ', mean_shift_value)
        
        if mean_shift_value == 0:
            stop[i] = True
        #print(stop)
        #update the center
        kcenters[i] = [x_mean, y_mean]
    
    if disp:
        print('...................................')


    #check if all the cluster centers have converged
    if sum(stop) == k:
        print('all the {} cluster centers have converged on the {}th iteration'.format(k, n))
        exit()



'''
data = np.array([
    [1, 2],
    [2, 3],
    [3, 6],
])
x, y = data.T
plt.scatter(x,y)
'''

'''
def save_plt(m,c,epoch_id):
    
    
    # Plot
    plt.scatter(x_list, y_list, color = 'cyan')#, s=area, c=colors, alpha=0.5)
    plt.title('fake data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    #plt.show()

    x_temp = x_max - x_min
    line_coord1 = [x_min , x_max ]
    line_coord2 = [x_min*m_fake + c_fake, m_fake*x_max + c_fake]
    plt.plot(line_coord1, line_coord2 , color = 'green')#, 'k-')    
    
    line_coord1 = [x_min , x_max]
    line_coord2 = [x_min*m + c , m*x_max + c]
    plt.plot(line_coord1, line_coord2 , color = 'pink')#, 'k-')
    plt.savefig('result/res{}.png'.format(epoch_id))
    #plt.show()    
'''
