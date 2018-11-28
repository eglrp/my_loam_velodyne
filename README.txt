####gedit ~/.bashrc

roslaunch loam_velodyne loam_velodyne.launch

rosbag record -o out /laser_cloud_surround 


rosrun pcl_ros bag_to_pcd out_2018-09-28-22-16-26.bag /laser_cloud_surround pcd

