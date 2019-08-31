## MeanShift Clustering

Mean shift clustering is a sliding-window-based algorithm that attempts to find dense areas of data points. It is a centroid-based algorithm meaning that the goal is to locate the center points of each group/class, which works by updating candidates for center points to be the mean of the points within the sliding-window. These candidate windows are then filtered in a post-processing stage to eliminate near-duplicates, forming the final set of center points and their corresponding groups.

In contrast to K-means clustering there is no need to select the number of clusters as mean-shift automatically discovers this. That’s a massive advantage. The fact that the cluster centers converge towards the points of maximum density is also quite desirable as it is quite intuitive to understand and fits well in a naturally data-driven sense. 

The drawback is that the selection of the window size/radius “r” can be non-trivial.

More on Clustering: https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68

### Generated Data
![](MeanShift/results/0*generated_data.jpg)

### Actual Cluster Centers
![](MeanShift/results/2*cluster_centers.jpg)

### Predicting Cluster Centers

* #### Initialization of Predicted Cluster Centers
  ![](MeanShift/results/1*progress_0000_initialization_of_cluster_centers.jpg)

* #### Predicted Cluster Centers after 2nd Iteration
  ![](MeanShift/results/1*progress_0002_cluster_centers.jpg)

* #### Predicted Cluster Centers after 4th Iteration
  ![](MeanShift/results/1*progress_0004_cluster_centers.jpg)

* #### Predicted Cluster Centers after 6th Iteration
  ![](MeanShift/results/1*progress_0006_cluster_centers.jpg)

### Final Mean Shift Clustering Result
![](MeanShift/results/3*meanShift_result.jpg)
