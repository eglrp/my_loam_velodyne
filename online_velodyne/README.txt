source devel/setup.bash

加载： roslaunch velodyne_pointcloud VLP16_points.launch calibration:=/home/zhao/desktop/loam_velodyne_project/online_velodyne/VLP-16.yaml

实时显示点云图：rviz -f velodyne

记录数据：rosbag record -O out /velodyne_points


