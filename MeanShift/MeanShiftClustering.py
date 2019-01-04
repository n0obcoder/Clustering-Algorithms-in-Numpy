__author__ = 'Animesh'

#NOTE: Run the code and hit 'Q' whenever a matplotlib plot pops up.

import numpy as np
import cv2
import os,sys,glob,pdb
from matplotlib import pyplot as plt
import matplotlib
from collections import OrderedDict

iterations = 1000
scatterplot_size = 25
disp = True
color_list = ['orange',  'lime','violet','pink','cornflowerblue','deepskyblue','lawngreen','red','cyan','purple','black'] #https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
###
#numpy.random.normal(loc=0.0, scale=1.0, size=None)
###
############################# DATA GENERATION #############################
def gen_data(data_points,loc):
    #print('data_points: ', data_points)
    #print('loc: ', loc)
    x_center = loc[0]
    y_center = loc[1]
    x = np.round(np.random.normal(size= data_points, loc= x_center)*100, 0)
    #print(x)
    #exit()
    y = np.round(np.random.normal(size= data_points, loc= y_center)*100, 0)
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
x, y = gen_data(100,[8,5]) #plt.hist(x) shows that is a normal distribution
plt.scatter(x, y, s= scatterplot_size)
x_data = np.hstack((x_data, x))
y_data = np.hstack((y_data, y))
#...................................
x, y = gen_data(50,[-2,-2]) 
plt.scatter(x, y, s= scatterplot_size)
x_data = np.hstack((x_data, x))
y_data = np.hstack((y_data, y))
#...................................
x, y = gen_data(80,[-5,10]) 
plt.scatter(x, y, s= scatterplot_size)
x_data = np.hstack((x_data, x))
y_data = np.hstack((y_data, y))
##...................................
x, y = gen_data(100,[10,-5]) 
plt.scatter(x, y, s= scatterplot_size)
x_data = np.hstack((x_data, x))
y_data = np.hstack((y_data, y))
#...................................
x, y = gen_data(300,[2,3]) 
plt.scatter(x, y, s= scatterplot_size)
x_data = np.hstack((x_data, x))
y_data = np.hstack((y_data, y))
#...................................
plt.title('2D data')
plt.xlabel('Some Feature x')
plt.ylabel('Some Feature y')
plt.grid()
plt.savefig(results_dir + '/0*generated_data.jpg')
plt.show()
print('>>>>>>>>>>>>>>>>>>>>>>>>>> DATA POINTS GENERATED')

############################# Mean Shift Clustering #############################

#find out the extremes of our data space

x_min= np.min(x_data)
x_max= np.max(x_data)
y_min= np.min(y_data)
y_max= np.max(y_data)

x_span = x_max - x_min
y_span = y_max - y_min
print('x_min, x_max, y_min, y_max: ', x_min, x_max, y_min, y_max)
print('x_span, y_span: ', x_span, y_span)

#initializing random cluster centers
num_centers = 15
x_centers = np.linspace(x_min, x_max, num=num_centers)
y_centers = np.linspace(y_min, y_max, num=num_centers)

centers = []
cluster_counter = 0
for x_c in range(len(x_centers)):
    for y_c in range(len(y_centers)):
        center = [np.round(x_centers[x_c], 2), np.round(y_centers[y_c], 2)]
        print('initializing cluster center #{} at {}'.format(cluster_counter, center))
        centers.append(center)
        cluster_counter += 1

#exit()
#print('initialized centers are: ', centers)

def display_data_points():
    plt.scatter(x_data, y_data, color = 'gray', s= scatterplot_size)

def display_centers(centers):
    for i in range(len(centers)):
        plt.scatter(centers[i][0], centers[i][1], color='Red', s= scatterplot_size)

#displaying initialized cluster centers
initial_data_point_color = 'Gray' 
plt.scatter(x_data, y_data, color= initial_data_point_color, s= scatterplot_size)
display_centers(centers)
plt.title('Initialization of Cluster Centers')
plt.xlabel('Some Feature x')
plt.ylabel('Some Feature y')
plt.grid()
plt.savefig(results_dir + '/1*progress_{}_initialization_of_cluster_centers.jpg'.format(format(str(0).zfill(len(str(iterations))))))
plt.show()

#exit()
#To explain mean-shift we will consider a set of points in two-dimensional space like the above illustration. We begin with a circular sliding window centered at a point C (randomly selected) and having radius r as the kernel. Mean shift is a hill climbing algorithm which involves shifting this kernel iteratively to a higher density region on each step until convergence.

# define radius of the kernel and minPoints
r = 300 #highre value of r is important. otherwise there might exist cluster centers very very close to each other. by rounding them off to an integer, the gap reduces to zero and we can extract unique cluster centers.
minPoints = 10
interCLuster_dist = 10
#lets find out which data points fall under the radius of which cluster centers. 
#NOTE: a data point can fall under multiple cluster points.

#define a dictionary that would contain all the information of the data points falling under the specified radius from each of the initialized cluster centers. 
points_in_clusters = OrderedDict()
converged = [False]*cluster_counter
show_centers = [True]*cluster_counter

def does_it_fall_under(x,y,center):
    dist = np.sqrt(np.square(x - center[0]) + np.square(y - center[1]))
    return dist <= r
    
def get_mean_shift_dist(old_center, new_center):
    dist = np.sqrt(np.square(old_center[0] - new_center[0]) + np.square(old_center[1] - new_center[1]))
    return dist

def get_dist(pointA, pointB):
    dist = np.sqrt(np.square(pointA[0] - pointB[0]) + np.square(pointA[1] - pointB[1]))
    return dist

def get_final_clusters(centers, points_in_clusters_, show_centers):
    
    #print(len(centers)) #100
    #print(len(points_in_clusters)) #<100
    
    points_in_clusters = points_in_clusters_
    
    #'''
    print('================================================entering')
    while True:
        #print('WATCHOUT >>>', len(points_in_clusters))
        print('sum(show_centers): ' , sum(show_centers))
        brk = False
        #iterate over all the cluster centers
        for i in range(len(centers)):
            #iterate over all the cluster centers again
            for j in range(len(centers)):
                temp_point_list = []

                if (i != j) and ('cluster{}'.format(i) in points_in_clusters) and ('cluster{}'.format(j) in points_in_clusters):
                    #print(centers[i], centers[j], get_dist(centers[i], centers[j]) )
                    if get_dist(centers[i], centers[j]) < interCLuster_dist:
                    
                        #append all the unique data points falling under the cluster center with indices i and j
                        for k in points_in_clusters['cluster{}'.format(i)]:
                            temp_point_list.append(k)
                        for l in points_in_clusters['cluster{}'.format(j)]:
                            if l not in temp_point_list:
                                temp_point_list.append(l)
                                
                        #set the centers[i] to the mean of the points in the temp_point_list
                        temp_arr = np.array(temp_point_list)
                        mean = np.round(np.mean(temp_arr, axis= 0), 0)
                        centers[i][0] = mean[0]
                        centers[i][1] = mean[1]
                        
                        #del points_in_clusters['cluster{}'.format(j)]
                        del points_in_clusters['cluster{}'.format(j)]

                        #set show_centers[i] = False
                        show_centers[j] = False

                        #print('GETDIST >>>>> ', get_dist(centers[i], centers[j]))
                        #print('deleted points in clusters')
                        #print('WATCH >>>', len(points_in_clusters))
                        #print('sum(show_centers): ' , sum(show_centers))
                        p()                        
                        brk = True
                        break
    
            if brk == True:
                break
        
        if not brk:
            break
                
    #pdb.set_trace()
    #'''    
    print('fin ')
    return points_in_clusters, show_centers
    
def visualize_clusters(final_centers):
    #iterating over all the data points
    for i in range(len(x_data)):
        x = x_data[i]
        y = y_data[i]
        #print('x,y: ',x,y)
        dist_from_clusters = []
        
        #creating a np array with shape same as the shape of final_centers
        a_data_point_array = np.ones(final_centers.shape)
        #filling its 1st column with x and 2nd column with y
        a_data_point_array[:, 0] = x
        a_data_point_array[:, 1] = y
        diff = a_data_point_array - final_centers
        #print('diff.shape: ', diff.shape)
        dist = np.sqrt(np.square(diff[:,0]) + np.square(diff[:,1]))
        #print('dist.shape: ', dist.shape)
        cluster_num = np.argmin(dist)
        #print('cluster_num: ', cluster_num)
        plt.scatter(x, y, color = color_list[cluster_num], s= scatterplot_size)
        
    #iterating over all the final clusters
    for i in range(final_centers.shape[0]):
        cluster_center = final_centers[i, :]
        #print('cluster_center: ', cluster_center)
        plt.scatter(cluster_center[0], cluster_center[1], color = 'red', s= scatterplot_size)
    
    plt.title('Mean Shift Clustering Result')
    plt.xlabel('Some Feature x')
    plt.ylabel('Some Feature y')
    plt.grid()
    plt.savefig(results_dir + '/3*meanShift_result.jpg')
    plt.show()
    
    print('job done i guess !!! :D')
        #exit()

def p():
    print('-------------------------------')

# run the mean shift clustering algo
for itr in range(iterations):
    print
    print('                  *******************   ITERATION #{}   *******************'.format(itr+1))
    
    #iterarting over all the cluster centers
    for c in range(len(centers)):

        if converged[c] == True:
            #print('cluster center #{} has previously been converged'.format(c))
            continue
        
        print('searching for the data points that fall under the cluster center #{} that is {}'.format(c, centers[c]))
        #exit()
        #add an list item in the points_in_clusters dictionary
        points_in_clusters['cluster{}'.format(c)] = []
        
        points_within = 0
        #iterating over all the data points
        for i in range(len(x_data)):
            x= x_data[i]
            y= y_data[i]
            
            fall_under = does_it_fall_under(x,y,centers[c])
            if fall_under == True:
                points_in_clusters['cluster{}'.format(c)].append([x,y])
                #print('[{},{}] added'.format(x,y))
                points_within += 1
        
        #make sure that the total number of points falling under the radius of this cluster center should be >= minPoints
        if points_within < minPoints:
            #print('DELETING CLUSTER CENTER: only {} data point(s) found falling under the cluster center#{}'.format(points_within, c))
            #delete the cluster information
            del points_in_clusters['cluster{}'.format(c)]
            #set the convergance flag to True
            converged[c] = True
            show_centers[c] = False
            #we can more on to the next cluster center now
            continue
        
        #we reached till this point because the convergence flag is at False and also the number of data points falling under the radius of this cluster center >= minPoints
        
        print('we have {} data points falling under this cluster center#{}'.format(points_within, c))
        #now lets calculate the mean of all the data points falling under this cluster center
        data_points_arr = np.array(points_in_clusters['cluster{}'.format(c)])
        #print(data_points_arr)
        
        
        ##TODO
        ##lets visualize the current cluster center and its circular span 
        #patch= matplotlib.patches.Circle((centers[c][0], centers[c][1]), radius= r)
        #plt.scatter(centers[c][0], centers[c][1], color = 'Black')
        #display_data_points()
        #ax=plt.gca()
        #ax.add_patch(patch)
        #plt.show()
        #exit()
        
        mean = np.round(np.mean(data_points_arr, axis= 0), 0) #rounding this mean is very important. otherwise there might exist cluster centers very very close to each other. by rounding them off to an integer, the gap reduces to zero and we can extract unique cluster centers.
        #print('mean: ', mean)
        #print(mean[0], mean[1])

        #calculate the distance between the current and the new cluster centers
        mean_shift_dist = get_mean_shift_dist(centers[c], mean)
        print('mean_shift_dist: ', mean_shift_dist)
        if mean_shift_dist < 0.005:  #mean shift threshold
            #print('CONVERGED')
            converged[c] = True
            #p()
            continue
        else:
            #print('shifting the cluster center')
            centers[c][0] = mean[0]
            centers[c][1] = mean[1]
            
        #print('converged: ', converged)
        #p()
    
    #print('len(points_in_clusters):',len(points_in_clusters))
    #print('sum(show_centers): ', sum(show_centers))
    
    #TODO
    #filter out the cluster centers if they they are very close to another cluster center
    points_in_clusters, show_centers = get_final_clusters(centers, points_in_clusters, show_centers)
    
    
    #visualizing the progress of the Algorithm
    display_data_points()
    #pdb.set_trace()
    display_centers(centers)
    plt.title('Clustering @iter {}'.format(itr+1))
    plt.xlabel('Some Feature x')
    plt.ylabel('Some Feature y') 
    plt.grid()
    plt.savefig(results_dir + '/1*progress_{}_cluster_centers.jpg'.format(str(itr+1).zfill(len(str(iterations)))))
    plt.show()
    #exit()
    #check if all the cluster points have converged
    if sum(converged) == cluster_counter:
        #print('all the {} cluster centers have converged on the iteration #{}'.format(cluster_counter, itr))
        #print('following are the data points falling under their respective cluster centers')
        
        for key, value in points_in_clusters.iteritems():
            
            print('>>> ',key)
            #print(value)
            temp_c = int(key[7:])
            print('number of points in this cluster center#{} with center at {}: {}'.format(temp_c, centers[temp_c] , len(value)))
            print('_____________')
        #print('total number of clusters found', len(points_in_clusters))
        
        print('iterations: {}/{}'.format(itr,iterations))
        print('r: ', r)
        print('minPoints: ', minPoints)
        print('sum(show_centers): ', sum(show_centers))
        print('len(points_in_clusters): ', len(points_in_clusters))
        print('len(centers) started with: ', len(centers))
        print('interCLuster_dist: ', interCLuster_dist)
        
        #'''
        #visualizing cluster centers
        display_data_points()
        final_centers = []
        for idx, show_cen in enumerate(show_centers):
            if show_cen:
                plt.scatter(centers[idx][0], centers[idx][1], color = 'red', s= scatterplot_size)
                #print(idx)
                final_centers.append(centers[idx])

        final_centers = np.array(final_centers)
        final_centers = np.unique(final_centers,axis = 0)
        #print(np.array(final_centers))
        print('=====> {} unique cluster centers are found'.format(final_centers.shape[0]))
        print('the unique centers are: ', final_centers)
        plt.title('Cluster Centers')
        plt.xlabel('Some Feature x')
        plt.ylabel('Some Feature y')
        plt.grid()
        plt.savefig(results_dir + '/2*cluster_centers.jpg')
        plt.show()
        #'''
        
        #visualizing clusters
        visualize_clusters(final_centers)


        exit()

    #print('points_in_clusters: ', points_in_clusters)
    #exit()