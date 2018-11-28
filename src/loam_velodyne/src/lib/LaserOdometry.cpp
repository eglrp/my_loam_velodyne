
#include "loam_velodyne/LaserOdometry.h"
#include "loam_velodyne/common.h"
#include "math_utils.h"

#include <pcl/filters/filter.h>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>


namespace loam {

using std::sin;
using std::cos;
using std::asin;
using std::atan2;
using std::sqrt;
using std::fabs;
using std::pow;


LaserOdometry::LaserOdometry(const float& scanPeriod,
                             const uint16_t& ioRatio,
                             const size_t& maxIterations)
      : _scanPeriod(scanPeriod),
        _ioRatio(ioRatio),
        _systemInited(false),
        _frameCount(0),
        _maxIterations(maxIterations),
        _deltaTAbort(0.1),
        _deltaRAbort(0.1),
        _cornerPointsSharp(new pcl::PointCloud<pcl::PointXYZI>()),
        _cornerPointsLessSharp(new pcl::PointCloud<pcl::PointXYZI>()),
        _surfPointsFlat(new pcl::PointCloud<pcl::PointXYZI>()),
        _surfPointsLessFlat(new pcl::PointCloud<pcl::PointXYZI>()),
        _laserCloud(new pcl::PointCloud<pcl::PointXYZI>()),
        _lastCornerCloud(new pcl::PointCloud<pcl::PointXYZI>()),
        _lastSurfaceCloud(new pcl::PointCloud<pcl::PointXYZI>()),
        _laserCloudOri(new pcl::PointCloud<pcl::PointXYZI>()),
        _coeffSel(new pcl::PointCloud<pcl::PointXYZI>())
{
  // initialize odometry and odometry tf messages
  _laserOdometryMsg.header.frame_id = "/camera_init";
  _laserOdometryMsg.child_frame_id = "/laser_odom";

  _laserOdometryTrans.frame_id_ = "/camera_init";
  _laserOdometryTrans.child_frame_id_ = "/laser_odom";
}



bool LaserOdometry::setup(ros::NodeHandle &node,ros::NodeHandle &privateNode)
{
  // fetch laser odometry params
  float fParam;
  int iParam;

  if (privateNode.getParam("scanPeriod", fParam))  //读取launch文件中的param
  {
    if (fParam <= 0) {
      ROS_ERROR("Invalid scanPeriod parameter: %f (expected > 0)", fParam);
      return false;
    } else {
      _scanPeriod = fParam;
      ROS_INFO("Set scanPeriod: %g", fParam);
    }
  }

  if (privateNode.getParam("ioRatio", iParam)) {
    if (iParam < 1) {
      ROS_ERROR("Invalid ioRatio parameter: %d (expected > 0)", iParam);
      return false;
    } else {
      _ioRatio = iParam;
      ROS_INFO("Set ioRatio: %d", iParam);
    }
  }

  if (privateNode.getParam("maxIterations", iParam)) {
    if (iParam < 1) {
      ROS_ERROR("Invalid maxIterations parameter: %d (expected > 0)", iParam);
      return false;
    } else {
      _maxIterations = iParam;
      ROS_INFO("Set maxIterations: %d", iParam);
    }
  }

  if (privateNode.getParam("deltaTAbort", fParam)) {
    if (fParam <= 0) {
      ROS_ERROR("Invalid deltaTAbort parameter: %f (expected > 0)", fParam);
      return false;
    } else {
      _deltaTAbort = fParam;
      ROS_INFO("Set deltaTAbort: %g", fParam);
    }
  }

  if (privateNode.getParam("deltaRAbort", fParam)) {
    if (fParam <= 0) {
      ROS_ERROR("Invalid deltaRAbort parameter: %f (expected > 0)", fParam);
      return false;
    } else {
      _deltaRAbort = fParam;
      ROS_INFO("Set deltaRAbort: %g", fParam);
    }
  }


  // advertise laser odometry topics
  _pubLaserCloudCornerLast = node.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2);
  _pubLaserCloudSurfLast   = node.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2);
  _pubLaserCloudFullRes    = node.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 2);
  _pubLaserOdometry        = node.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 5);  //odometry topics


  // subscribe to scan registration topics
  _subCornerPointsSharp = node.subscribe<sensor_msgs::PointCloud2>
      ("/laser_cloud_sharp", 2, &LaserOdometry::laserCloudSharpHandler, this);

  _subCornerPointsLessSharp = node.subscribe<sensor_msgs::PointCloud2>
      ("/laser_cloud_less_sharp", 2, &LaserOdometry::laserCloudLessSharpHandler, this);

  _subSurfPointsFlat = node.subscribe<sensor_msgs::PointCloud2>
      ("/laser_cloud_flat", 2, &LaserOdometry::laserCloudFlatHandler, this);

  _subSurfPointsLessFlat = node.subscribe<sensor_msgs::PointCloud2>
      ("/laser_cloud_less_flat", 2, &LaserOdometry::laserCloudLessFlatHandler, this);

  _subLaserCloudFullRes = node.subscribe<sensor_msgs::PointCloud2>
      ("/velodyne_cloud_2", 2, &LaserOdometry::laserCloudFullResHandler, this);

  _subImuTrans = node.subscribe<sensor_msgs::PointCloud2>
      ("/imu_trans", 5, &LaserOdometry::imuTransHandler, this);

  return true;
}


/*
当前点云中的点相对第一个点去除因匀速运动产生的畸变，效果相当于得到在点云扫描开始位置静止扫描得到的点云
*/
void LaserOdometry::transformToStart(const pcl::PointXYZI& pi, pcl::PointXYZI& po)
{
    //插值系数计算，云中每个点的相对时间/点云周期10
  float s = 10 * (pi.intensity - int(pi.intensity));
  //线性插值：根据每个点在点云中的相对位置关系，乘以相应的旋转平移系数
  po.x = pi.x - s * _transform.pos.x();
  po.y = pi.y - s * _transform.pos.y();
  po.z = pi.z - s * _transform.pos.z();
  po.intensity = pi.intensity;

  Angle rx = -s * _transform.rot_x.rad();
  Angle ry = -s * _transform.rot_y.rad();
  Angle rz = -s * _transform.rot_z.rad();
  rotateZXY(po, rz, rx, ry);
}


/*
将上一帧点云中的点相对结束位置去除因匀速运动产生的畸变，效果相当于得到在点云扫描结束位置静止扫描得到的点云
*/
size_t LaserOdometry::transformToEnd(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud)
{
  size_t cloudSize = cloud->points.size();

  for (size_t i = 0; i < cloudSize; i++) {
    pcl::PointXYZI& point = cloud->points[i];
  //插值系数计算
    float s = 10 * (point.intensity - int(point.intensity));

    point.x -= s * _transform.pos.x();
    point.y -= s * _transform.pos.y();
    point.z -= s * _transform.pos.z();
    point.intensity = int(point.intensity);

    Angle rx = -s * _transform.rot_x.rad();
    Angle ry = -s * _transform.rot_y.rad();
    Angle rz = -s * _transform.rot_z.rad();
    rotateZXY(point, rz, rx, ry);
    rotateYXZ(point, _transform.rot_y, _transform.rot_x, _transform.rot_z);

    point.x += _transform.pos.x() - _imuShiftFromStart.x();
    point.y += _transform.pos.y() - _imuShiftFromStart.y();
    point.z += _transform.pos.z() - _imuShiftFromStart.z();

    rotateZXY(point, _imuRollStart, _imuPitchStart, _imuYawStart);
    rotateYXZ(point, -_imuYawEnd, -_imuPitchEnd, -_imuRollEnd);
  }

  return cloudSize;
}


//利用IMU修正旋转量，根据起始欧拉角，当前点云的欧拉角修正
void LaserOdometry::pluginIMURotation(const Angle& bcx, const Angle& bcy, const Angle& bcz,
                                      const Angle& blx, const Angle& bly, const Angle& blz,
                                      const Angle& alx, const Angle& aly, const Angle& alz,
                                      Angle &acx, Angle &acy, Angle &acz)
{
  float sbcx = bcx.sin();
  float cbcx = bcx.cos();
  float sbcy = bcy.sin();
  float cbcy = bcy.cos();
  float sbcz = bcz.sin();
  float cbcz = bcz.cos();

  float sblx = blx.sin();
  float cblx = blx.cos();
  float sbly = bly.sin();
  float cbly = bly.cos();
  float sblz = blz.sin();
  float cblz = blz.cos();

  float salx = alx.sin();
  float calx = alx.cos();
  float saly = aly.sin();
  float caly = aly.cos();
  float salz = alz.sin();
  float calz = alz.cos();

  float srx = -sbcx*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly)
            - cbcx*cbcz*(calx*saly*(cbly*sblz - cblz*sblx*sbly)
                         - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx)
            - cbcx*sbcz*(calx*caly*(cblz*sbly - cbly*sblx*sblz)
                         - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz);
  acx = -asin(srx);

  float srycrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*saly*(cbly*sblz - cblz*sblx*sbly)
                                               - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx)
                 - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz)
                                                 - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz)
                 + cbcx*sbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
  float crycrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz)
                                               - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz)
                 - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*saly*(cbly*sblz - cblz*sblx*sbly)
                                                 - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx)
                 + cbcx*cbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
  acy = atan2(srycrx / acx.cos(), crycrx / acx.cos());

  float srzcrx = sbcx*(cblx*cbly*(calz*saly - caly*salx*salz) - cblx*sbly*(caly*calz + salx*saly*salz) + calx*salz*sblx)
                 - cbcx*cbcz*((caly*calz + salx*saly*salz)*(cbly*sblz - cblz*sblx*sbly)
                              + (calz*saly - caly*salx*salz)*(sbly*sblz + cbly*cblz*sblx)
                              - calx*cblx*cblz*salz)
                 + cbcx*sbcz*((caly*calz + salx*saly*salz)*(cbly*cblz + sblx*sbly*sblz)
                              + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz)
                              + calx*cblx*salz*sblz);
  float crzcrx = sbcx*(cblx*sbly*(caly*salz - calz*salx*saly) - cblx*cbly*(saly*salz + caly*calz*salx) + calx*calz*sblx)
                 + cbcx*cbcz*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx)
                              + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly)
                              + calx*calz*cblx*cblz)
                 - cbcx*sbcz*((saly*salz + caly*calz*salx)*(cblz*sbly - cbly*sblx*sblz)
                              + (caly*salz - calz*salx*saly)*(cbly*cblz + sblx*sbly*sblz)
                              - calx*calz*cblx*sblz);
  acz = atan2(srzcrx / acx.cos(), crzcrx / acx.cos());
}


//相对于第一个点云即原点，积累旋转量
void LaserOdometry::accumulateRotation(Angle cx, Angle cy, Angle cz,
                                       Angle lx, Angle ly, Angle lz,
                                       Angle &ox, Angle &oy, Angle &oz)
{
  float srx = lx.cos()*cx.cos()*ly.sin()*cz.sin()
            - cx.cos()*cz.cos()*lx.sin()
            - lx.cos()*ly.cos()*cx.sin();
  ox = -asin(srx);

  float srycrx = lx.sin()*(cy.cos()*cz.sin() - cz.cos()*cx.sin()*cy.sin())
               + lx.cos()*ly.sin()*(cy.cos()*cz.cos() + cx.sin()*cy.sin()*cz.sin())
               + lx.cos()*ly.cos()*cx.cos()*cy.sin();
  float crycrx = lx.cos()*ly.cos()*cx.cos()*cy.cos()
               - lx.cos()*ly.sin()*(cz.cos()*cy.sin() - cy.cos()*cx.sin()*cz.sin())
               - lx.sin()*(cy.sin()*cz.sin() + cy.cos()*cz.cos()*cx.sin());
  oy = atan2(srycrx / ox.cos(), crycrx / ox.cos());

  float srzcrx = cx.sin()*(lz.cos()*ly.sin() - ly.cos()*lx.sin()*lz.sin())
               + cx.cos()*cz.sin()*(ly.cos()*lz.cos() + lx.sin()*ly.sin()*lz.sin())
               + lx.cos()*cx.cos()*cz.cos()*lz.sin();
  float crzcrx = lx.cos()*lz.cos()*cx.cos()*cz.cos()
               - cx.cos()*cz.sin()*(ly.cos()*lz.sin() - lz.cos()*lx.sin()*ly.sin())
               - cx.sin()*(ly.sin()*lz.sin() + ly.cos()*lz.cos()*lx.sin());
  oz = atan2(srzcrx / ox.cos(), crzcrx / ox.cos());
}



void LaserOdometry::laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr& cornerPointsSharpMsg)
{
  _timeCornerPointsSharp = cornerPointsSharpMsg->header.stamp;

  _cornerPointsSharp->clear();
  pcl::fromROSMsg(*cornerPointsSharpMsg, *_cornerPointsSharp);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*_cornerPointsSharp, *_cornerPointsSharp, indices);
  _newCornerPointsSharp = true;
}



void LaserOdometry::laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr& cornerPointsLessSharpMsg)
{
  _timeCornerPointsLessSharp = cornerPointsLessSharpMsg->header.stamp;

  _cornerPointsLessSharp->clear();
  pcl::fromROSMsg(*cornerPointsLessSharpMsg, *_cornerPointsLessSharp);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*_cornerPointsLessSharp, *_cornerPointsLessSharp, indices);
  _newCornerPointsLessSharp = true;
}



void LaserOdometry::laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr& surfPointsFlatMsg)
{
  _timeSurfPointsFlat = surfPointsFlatMsg->header.stamp;

  _surfPointsFlat->clear();
  pcl::fromROSMsg(*surfPointsFlatMsg, *_surfPointsFlat);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*_surfPointsFlat, *_surfPointsFlat, indices);
  _newSurfPointsFlat = true;
}



void LaserOdometry::laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr& surfPointsLessFlatMsg)
{
  _timeSurfPointsLessFlat = surfPointsLessFlatMsg->header.stamp;

  _surfPointsLessFlat->clear();
  pcl::fromROSMsg(*surfPointsLessFlatMsg, *_surfPointsLessFlat);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*_surfPointsLessFlat, *_surfPointsLessFlat, indices);
  _newSurfPointsLessFlat = true;
}


//接收全部点
void LaserOdometry::laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudFullResMsg)
{
  _timeLaserCloudFullRes = laserCloudFullResMsg->header.stamp;

  _laserCloud->clear();
  pcl::fromROSMsg(*laserCloudFullResMsg, *_laserCloud);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*_laserCloud, *_laserCloud, indices);
  _newLaserCloudFullRes = true;
}


//接收imu消息
void LaserOdometry::imuTransHandler(const sensor_msgs::PointCloud2ConstPtr& imuTransMsg)
{
  _timeImuTrans = imuTransMsg->header.stamp;

  pcl::PointCloud<pcl::PointXYZ> imuTrans;
  pcl::fromROSMsg(*imuTransMsg, imuTrans);

  //根据发来的消息提取imu信息
  _imuPitchStart = imuTrans.points[0].x;
  _imuYawStart = imuTrans.points[0].y;
  _imuRollStart = imuTrans.points[0].z;

  _imuPitchEnd = imuTrans.points[1].x;
  _imuYawEnd = imuTrans.points[1].y;
  _imuRollEnd = imuTrans.points[1].z;

  _imuShiftFromStart = imuTrans.points[2];
  _imuVeloFromStart = imuTrans.points[3];

  _newImuTrans = true;
}



void LaserOdometry::spin()
{
  ros::Rate rate(100);
  bool status = ros::ok();

  // loop until shutdown
  while (status) {
    ros::spinOnce();

    // try processing new data
    process();

    status = ros::ok();
    rate.sleep();
  }
}



void LaserOdometry::reset()
{    //消息接收标志
  _newCornerPointsSharp = false;
  _newCornerPointsLessSharp = false;
  _newSurfPointsFlat = false;
  _newSurfPointsLessFlat = false;
  _newLaserCloudFullRes = false;
  _newImuTrans = false;
}



bool LaserOdometry::hasNewData()
{  //同步作用，确保同时收到同一个点云的特征点以及IMU信息才进入
  return _newCornerPointsSharp && _newCornerPointsLessSharp && _newSurfPointsFlat &&
         _newSurfPointsLessFlat && _newLaserCloudFullRes && _newImuTrans &&
         fabs((_timeCornerPointsSharp - _timeSurfPointsLessFlat).toSec()) < 0.005 &&
         fabs((_timeCornerPointsLessSharp - _timeSurfPointsLessFlat).toSec()) < 0.005 &&
         fabs((_timeSurfPointsFlat - _timeSurfPointsLessFlat).toSec()) < 0.005 &&
         fabs((_timeLaserCloudFullRes - _timeSurfPointsLessFlat).toSec()) < 0.005 &&
         fabs((_timeImuTrans - _timeSurfPointsLessFlat).toSec()) < 0.005;
}



void LaserOdometry::process()
{
  if (!hasNewData()) {
    // waiting for new data to arrive...
    return;
  }

  // reset flags, etc.  消息接收标志
  reset();

  //将第一个点云数据集发送给laserMapping,从下一个点云数据开始处理
  if (!_systemInited) {
   //将cornerPointsLessSharp与laserCloudCornerLast交换,目的保存cornerPointsLessSharp的值下轮使用
    _cornerPointsLessSharp.swap(_lastCornerCloud);
   //将surfPointLessFlat与laserCloudSurfLast交换，目的保存surfPointsLessFlat的值下轮使用
    _surfPointsLessFlat.swap(_lastSurfaceCloud);

  //使用上一帧的特征点构建kd-tree
    _lastCornerKDTree.setInputCloud(_lastCornerCloud); //所有的边沿点集合
    _lastSurfaceKDTree.setInputCloud(_lastSurfaceCloud); //所有的平面点集合

    //记住原点的翻滚角和俯仰角
    _transformSum.rot_x += _imuPitchStart;
    _transformSum.rot_z += _imuRollStart;

    _systemInited = true;
    return;
  }

  pcl::PointXYZI coeff;
  bool isDegenerate = false;
  Eigen::Matrix<float,6,6> matP;

  _frameCount++;
  _transform.pos -= _imuVeloFromStart * _scanPeriod;


  size_t lastCornerCloudSize = _lastCornerCloud->points.size();
  size_t lastSurfaceCloudSize = _lastSurfaceCloud->points.size();

// 上一时刻特征边(曲率大)上的点云个数大于10， 特征面内的点云大于100  
// -> 保证足够多的特征点可用于t+1时刻的匹配
  if (lastCornerCloudSize > 10 && lastSurfaceCloudSize > 100) 
  {
    std::vector<int> pointSearchInd(1);
    std::vector<float> pointSearchSqDis(1);
    std::vector<int> indices;

    pcl::removeNaNFromPointCloud(*_cornerPointsSharp, *_cornerPointsSharp, indices);//// 剔除异常点
    size_t cornerPointsSharpNum = _cornerPointsSharp->points.size();// 当前时刻特征边上的点云个数
    size_t surfPointsFlatNum = _surfPointsFlat->points.size(); // 当前时刻特征面上的点云个数

    _pointSearchCornerInd1.resize(cornerPointsSharpNum);
    _pointSearchCornerInd2.resize(cornerPointsSharpNum);
    _pointSearchSurfInd1.resize(surfPointsFlatNum);
    _pointSearchSurfInd2.resize(surfPointsFlatNum);
    _pointSearchSurfInd3.resize(surfPointsFlatNum);

    //Levenberg-Marquardt算法(L-M method)，非线性最小二乘算法，最优化算法的一种   最多迭代25次
    for (size_t iterCount = 0; iterCount < _maxIterations; iterCount++) 
    {
        pcl::PointXYZI pointSel, pointProj, tripod1, tripod2, tripod3;
        _laserCloudOri->clear();
        _coeffSel->clear();

       /************  Paper Algorithm1:Ladar Odometry   *************/

       //edge points************************8
        for (int i = 0; i < cornerPointsSharpNum; i++) 
          {
            transformToStart(_cornerPointsSharp->points[i], pointSel); // 将点坐标转换到起始点云坐标系中

            if (iterCount % 5 == 0) // 每迭代五次,搜索一次最近点和次临近点(降采样)
            {
              pcl::removeNaNFromPointCloud(*_lastCornerCloud, *_lastCornerCloud, indices);
              //找到pointSel(当前时刻边特征中的某一点)在laserCloudCornerLast中的1个最邻近点 
              //边沿点未经过体素栅格滤波，一般边沿点本来就比较少，不做滤波
              // -> 返回pointSearchInd(点对应的索引)  pointSearchSqDis(pointSel与对应点的欧氏距离)
              _lastCornerKDTree.nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
              int closestPointInd = -1, minPointInd2 = -1;

              //寻找相邻线距离目标点距离最小的点
              //再次提醒：velodyne是2度一线，scanID相邻并不代表线号相邻，相邻线度数相差2度，也即线号scanID相差2   ??
              if (pointSearchSqDis[0] < 25) 
              {//找到的最近点距离的确很近的话
                closestPointInd = pointSearchInd[0];
                //提取最近点线号
                int closestPointScan = int(_lastCornerCloud->points[closestPointInd].intensity);

                float pointSqDis, minPointSqDis2 = 25; //初始门槛值5米，可大致过滤掉scanID相邻，但实际线不相邻的值

                //寻找距离目标点最近距离的平方和最小的点
                for (int j = closestPointInd + 1; j < cornerPointsSharpNum; j++) {//向scanID增大的方向查找
                  if (int(_lastCornerCloud->points[j].intensity) > closestPointScan + 2.5) //非相邻线
                    break;                  

                  pointSqDis = calcSquaredDiff(_lastCornerCloud->points[j], pointSel);

                  if (int(_lastCornerCloud->points[j].intensity) > closestPointScan) 
                  {//确保两个点不在同一条scan上（相邻线查找应该可以用scanID == closestPointScan +/- 1 来做）
                    if (pointSqDis < minPointSqDis2) {//距离更近，要小于初始值5米
                      minPointSqDis2 = pointSqDis;//更新最小距离与点序
                      minPointInd2 = j;
                    }
                  }
                }
                for (int j = closestPointInd - 1; j >= 0; j--) {//同理  向scanID减小的方向查找
                  if (int(_lastCornerCloud->points[j].intensity) < closestPointScan - 2.5) {
                    break;
                  }

                  pointSqDis = calcSquaredDiff(_lastCornerCloud->points[j], pointSel);

                  if (int(_lastCornerCloud->points[j].intensity) < closestPointScan) {
                    if (pointSqDis < minPointSqDis2) {
                      minPointSqDis2 = pointSqDis;
                      minPointInd2 = j;
                    }
                  }
                }
              }
              //记住组成线的点序
              _pointSearchCornerInd1[i] = closestPointInd; //kd-tree最近距离点，-1表示未找到满足的点
              _pointSearchCornerInd2[i] = minPointInd2;  //另一个最近的，-1表示未找到满足的点
            }
        //构建Jaccobian矩阵  特征线：
          if (_pointSearchCornerInd2[i] >= 0)  //大于等于0，不等于-1，说明两个点都找到了   
            {
              tripod1 = _lastCornerCloud->points[_pointSearchCornerInd1[i]];
              tripod2 = _lastCornerCloud->points[_pointSearchCornerInd2[i]];

              //选择的特征点记为O，kd-tree最近距离点记为A，另一个最近距离点记为B
              // 当前点云的点坐标  0
              float x0 = pointSel.x;
              float y0 = pointSel.y;
              float z0 = pointSel.z;
              // 上一时刻最邻近点的点坐标  A
              float x1 = tripod1.x;
              float y1 = tripod1.y;
              float z1 = tripod1.z;
              // 上一时刻次临近点的点坐标  B
              float x2 = tripod2.x;
              float y2 = tripod2.y;
              float z2 = tripod2.z;

             // 文章中公式(2)的分子部分->分别作差并叉乘后的向量模长
             //向量OA=(x0-x1, y0-y1, z0-z1),向量OB=(x0-x2,y0-y2,z0-z2)向量AB = （x1 - x2, y1 - y2, z1 - z2）
              //向量OA OB的向量积(即叉乘)为：
              //|  i      j      k  |
              //|x0-x1  y0-y1  z0-z1|
              //|x0-x2  y0-y2  z0-z2|
              //模为：
              float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))  
                                * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                  * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                                  * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

             // 文章中公式(2)分母部分 -> 两点间距离
              float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

              // 向量[la；lb；lc] 为距离ld2分别对x0 y0 z0的偏导
              //AB方向的单位向量与OAB平面的单位法向量的向量积在各轴上的分量（d的方向）
              //x轴分量i
              float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                          + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;
             //y轴分量j
              float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                          - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;
               //z轴分量k
              float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                          + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;
              //点到线的距离，d = |向量OA 叉乘 向量OB|/|AB|
              float ld2 = a012 / l12;

              // TODO: Why writing to a variable that's never read?
              pointProj = pointSel;
              pointProj.x -= la * ld2;
              pointProj.y -= lb * ld2;
              pointProj.z -= lc * ld2;

              //权重计算，距离越大权重越小，距离越小权重越大，得到的权重范围<=1
              float s = 1;  // 阻尼因子
              if (iterCount >= 5) {//5次迭代之后开始增加权重因素
                s = 1 - 1.8f * fabs(ld2); // 点到直线距离越小阻尼因子越大
              }
              //考虑权重
              coeff.x = s * la;
              coeff.y = s * lb;
              coeff.z = s * lc;
              coeff.intensity = s * ld2;

              if (s > 0.1 && ld2 != 0) {// 满足阈值(ld2 < 0.5)，将特征点插入 只保留权重大的，也即距离比较小的点，同时也舍弃距离为零的
                _laserCloudOri->push_back(_cornerPointsSharp->points[i]);
                _coeffSel->push_back(coeff);
              }
            }
          }


      //planar points***********************************
     //对本次接收到的曲率最小的点,从上次接收到的点云曲率比较小的点中找三点组成平面，一个使用kd-tree查找，
     //另外一个在同一线上查找满足要求的，第三个在不同线上查找满足要求的
        for (int i = 0; i < surfPointsFlatNum; i++) 
          {
            transformToStart(_surfPointsFlat->points[i], pointSel);

            if (iterCount % 5 == 0) {
             //kd-tree最近点查找，在经过体素栅格滤波之后的平面点中查找，一般平面点太多，滤波后最近点查找数据量小
              _lastSurfaceKDTree.nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
              int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
              if (pointSearchSqDis[0] < 25) {
                closestPointInd = pointSearchInd[0];
                int closestPointScan = int(_lastSurfaceCloud->points[closestPointInd].intensity);

                float pointSqDis, minPointSqDis2 = 25, minPointSqDis3 = 25;
                for (int j = closestPointInd + 1; j < surfPointsFlatNum; j++) {
                  if (int(_lastSurfaceCloud->points[j].intensity) > closestPointScan + 2.5) {
                    break;
                  }

                  pointSqDis = calcSquaredDiff(_lastSurfaceCloud->points[j], pointSel);

                  if (int(_lastSurfaceCloud->points[j].intensity) <= closestPointScan) 
                  {//如果点的线号小于等于最近点的线号(应该最多取等，也即同一线上的点)
                    if (pointSqDis < minPointSqDis2) {
                      minPointSqDis2 = pointSqDis;
                      minPointInd2 = j;
                    }
                  } 
                  else 
                  {//如果点处在大于该线上
                    if (pointSqDis < minPointSqDis3) {
                      minPointSqDis3 = pointSqDis;
                      minPointInd3 = j;
                    }
                  }
                }

                for (int j = closestPointInd - 1; j >= 0; j--) {  //同理
                  if (int(_lastSurfaceCloud->points[j].intensity) < closestPointScan - 2.5) {
                    break;
                  }

                  pointSqDis = calcSquaredDiff(_lastSurfaceCloud->points[j], pointSel);

                  if (int(_lastSurfaceCloud->points[j].intensity) >= closestPointScan) {
                    if (pointSqDis < minPointSqDis2) {
                      minPointSqDis2 = pointSqDis;
                      minPointInd2 = j;
                    }
                  } else {
                    if (pointSqDis < minPointSqDis3) {
                      minPointSqDis3 = pointSqDis;
                      minPointInd3 = j;
                    }
                  }
                }
              }

              _pointSearchSurfInd1[i] = closestPointInd; //kd-tree最近距离点,-1表示未找到满足要求的点
              _pointSearchSurfInd2[i] = minPointInd2;  //同一线号上的距离最近的点，-1表示未找到满足要求的点
              _pointSearchSurfInd3[i] = minPointInd3;  //不同线号上的距离最近的点，-1表示未找到满足要求的点
            }

          //构建Jaccobian矩阵  特征点：
            if (_pointSearchSurfInd2[i] >= 0 && _pointSearchSurfInd3[i] >= 0)  //找到了三个点
            {
              tripod1 = _lastSurfaceCloud->points[_pointSearchSurfInd1[i]];//A点
              tripod2 = _lastSurfaceCloud->points[_pointSearchSurfInd2[i]];//B点
              tripod3 = _lastSurfaceCloud->points[_pointSearchSurfInd3[i]];//D点

            // 向量[pa；pb；pc] = 点到面的距离对x0 y0 z0的偏导
              //向量AB AC的向量积（即叉乘），得到的是法向量
              //x轴方向分向量i
              float pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z)
                        - (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z);
              float pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x)
                        - (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x);
              float pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y)
                        - (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y);
              float pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);

              //法向量的模
              float ps = sqrt(pa * pa + pb * pb + pc * pc);
             //pa pb pc为法向量各方向上的单位向量
              pa /= ps;
              pb /= ps;
              pc /= ps;
              pd /= ps;

              //点到面的距离：向量OA与与法向量的点积除以法向量的模
              float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

              // TODO: Why writing to a variable that's never read? Maybe it should be used afterwards?
              //unused

              pointProj = pointSel;
              pointProj.x -= pa * pd2;
              pointProj.y -= pb * pd2;
              pointProj.z -= pc * pd2;

             //同理计算权重
              float s = 1;
              if (iterCount >= 5) {
                s = 1 - 1.8f * fabs(pd2) / sqrt(calcPointDistance(pointSel));
              }

             //考虑权重
              coeff.x = s * pa;
              coeff.y = s * pb;
              coeff.z = s * pc;
              coeff.intensity = s * pd2;

              if (s > 0.1 && pd2 != 0) {//保存原始点与相应的系数
                _laserCloudOri->push_back(_surfPointsFlat->points[i]);
                _coeffSel->push_back(coeff);
              }
            }
          }  //end of planar points

         //这就是LM的解算过程，具体程序实现：
          int pointSelNum = _laserCloudOri->points.size();  // 匹配到的点的个数(即存在多少个约束) 
          if (pointSelNum < 10)      //满足要求的特征点至少10个，特征匹配数量太少弃用此帧数据
            continue;
          
          Eigen::Matrix<float,Eigen::Dynamic,6> matA(pointSelNum, 6);
          Eigen::Matrix<float,6,Eigen::Dynamic> matAt(6,pointSelNum);
          Eigen::Matrix<float,6,6> matAtA;
          Eigen::VectorXf matB(pointSelNum);
          Eigen::Matrix<float,6,1> matAtB;
          Eigen::Matrix<float,6,1> matX;

          //计算matA,matB矩阵
          for (int i = 0; i < pointSelNum; i++) 		  // 构建Jaccobian矩阵
          {
              /**
            * 采用Levenberg-Marquardt计算
            * 首先建立当前时刻Lidar坐标系下提取到的特征点与点到直线/平面
            * 的约束方程。而后对约束方程求对坐标变换(3旋转+3平移)的偏导
            * 公式参见论文(2)-(8)
            */
              const pcl::PointXYZI& pointOri = _laserCloudOri->points[i]; // 当前时刻点坐标
              coeff = _coeffSel->points[i];  // 该点所对应的偏导数

              float s = 1;

              float srx = sin(s * _transform.rot_x.rad());
              float crx = cos(s * _transform.rot_x.rad());
              float sry = sin(s * _transform.rot_y.rad());
              float cry = cos(s * _transform.rot_y.rad());
              float srz = sin(s * _transform.rot_z.rad());
              float crz = cos(s * _transform.rot_z.rad());
              float tx = s * _transform.pos.x();
              float ty = s * _transform.pos.y();
              float tz = s * _transform.pos.z();

              float arx = (-s*crx*sry*srz*pointOri.x + s*crx*crz*sry*pointOri.y + s*srx*sry*pointOri.z
                          + s*tx*crx*sry*srz - s*ty*crx*crz*sry - s*tz*srx*sry) * coeff.x
                          + (s*srx*srz*pointOri.x - s*crz*srx*pointOri.y + s*crx*pointOri.z
                            + s*ty*crz*srx - s*tz*crx - s*tx*srx*srz) * coeff.y
                          + (s*crx*cry*srz*pointOri.x - s*crx*cry*crz*pointOri.y - s*cry*srx*pointOri.z
                            + s*tz*cry*srx + s*ty*crx*cry*crz - s*tx*crx*cry*srz) * coeff.z;

              float ary = ((-s*crz*sry - s*cry*srx*srz)*pointOri.x
                          + (s*cry*crz*srx - s*sry*srz)*pointOri.y - s*crx*cry*pointOri.z
                          + tx*(s*crz*sry + s*cry*srx*srz) + ty*(s*sry*srz - s*cry*crz*srx)
                          + s*tz*crx*cry) * coeff.x
                          + ((s*cry*crz - s*srx*sry*srz)*pointOri.x
                            + (s*cry*srz + s*crz*srx*sry)*pointOri.y - s*crx*sry*pointOri.z
                            + s*tz*crx*sry - ty*(s*cry*srz + s*crz*srx*sry)
                            - tx*(s*cry*crz - s*srx*sry*srz)) * coeff.z;

              float arz = ((-s*cry*srz - s*crz*srx*sry)*pointOri.x + (s*cry*crz - s*srx*sry*srz)*pointOri.y
                          + tx*(s*cry*srz + s*crz*srx*sry) - ty*(s*cry*crz - s*srx*sry*srz)) * coeff.x
                          + (-s*crx*crz*pointOri.x - s*crx*srz*pointOri.y
                            + s*ty*crx*srz + s*tx*crx*crz) * coeff.y
                          + ((s*cry*crz*srx - s*sry*srz)*pointOri.x + (s*crz*sry + s*cry*srx*srz)*pointOri.y
                            + tx*(s*sry*srz - s*cry*crz*srx) - ty*(s*crz*sry + s*cry*srx*srz)) * coeff.z;

              float atx = -s*(cry*crz - srx*sry*srz) * coeff.x + s*crx*srz * coeff.y
                          - s*(crz*sry + cry*srx*srz) * coeff.z;

              float aty = -s*(cry*srz + crz*srx*sry) * coeff.x - s*crx*crz * coeff.y
                          - s*(sry*srz - cry*crz*srx) * coeff.z;

              float atz = s*crx*sry * coeff.x - s*srx * coeff.y - s*crx*cry * coeff.z;

              float d2 = coeff.intensity;

              matA(i, 0) = arx;
              matA(i, 1) = ary;
              matA(i, 2) = arz;
              matA(i, 3) = atx;
              matA(i, 4) = aty;
              matA(i, 5) = atz;
              matB(i, 0) = -0.05 * d2;
          }

        // 最小二乘计算(QR分解法)
          matAt = matA.transpose();
          matAtA = matAt * matA;
          matAtB = matAt * matB;
          //求解matAtA * matX = matAtB
          matX = matAtA.colPivHouseholderQr().solve(matAtB);

          if (iterCount == 0) 
          {
              Eigen::Matrix<float,1,6> matE;//特征值1*6矩阵
              Eigen::Matrix<float,6,6> matV;//特征向量6*6矩阵
              Eigen::Matrix<float,6,6> matV2;

              Eigen::SelfAdjointEigenSolver< Eigen::Matrix<float,6, 6> > esolver(matAtA);//计算矩阵的特征向量E及特征向量的反对称阵V
              matE = esolver.eigenvalues().real();
              matV = esolver.eigenvectors().real();

              matV2 = matV;

              isDegenerate = false;
             //特征值取值门槛
              float eignThre[6] = {10, 10, 10, 10, 10, 10};
              for (int i = 5; i >= 0; i--) 
              { //从小到大查找
                if (matE(0, i) < eignThre[i]) 
                {//特征值太小，则认为处在兼并环境中，发生了退化
                  for (int j = 0; j < 6; j++) 
                  {//对应的特征向量置为0
                    matV2(i, j) = 0;
                  }
                  isDegenerate = true;  // 存在比10小的特征值则出现退化
                } 
                else 
                  break;
                
              }
              //计算P矩阵
              matP = matV.inverse() * matV2;
          }

          if (isDegenerate)  //如果发生退化，只使用预测矩阵P计算
          {
            Eigen::Matrix<float,6,1> matX2;
            matX2 = matX;
            matX = matP * matX2;
          }

         //累加每次迭代的旋转平移量
          _transform.rot_x = _transform.rot_x.rad() + matX(0, 0);
          _transform.rot_y = _transform.rot_y.rad() + matX(1, 0);
          _transform.rot_z = _transform.rot_z.rad() + matX(2, 0);
          _transform.pos.x() += matX(3, 0);
          _transform.pos.y() += matX(4, 0);
          _transform.pos.z() += matX(5, 0);

          if( !pcl_isfinite(_transform.rot_x.rad()) ) _transform.rot_x = Angle();
          if( !pcl_isfinite(_transform.rot_y.rad()) ) _transform.rot_y = Angle();
          if( !pcl_isfinite(_transform.rot_z.rad()) ) _transform.rot_z = Angle();

          if( !pcl_isfinite(_transform.pos.x()) ) _transform.pos.x() = 0.0; //判断是否非数字
          if( !pcl_isfinite(_transform.pos.y()) ) _transform.pos.y() = 0.0;
          if( !pcl_isfinite(_transform.pos.z()) ) _transform.pos.z() = 0.0;

          //计算旋转平移量，如果很小就停止迭代
          float deltaR = sqrt(pow(rad2deg(matX(0, 0)), 2) +
                              pow(rad2deg(matX(1, 0)), 2) +
                              pow(rad2deg(matX(2, 0)), 2));
          float deltaT = sqrt(pow(matX(3, 0) * 100, 2) +
                              pow(matX(4, 0) * 100, 2) +
                              pow(matX(5, 0) * 100, 2));

          if (deltaR < _deltaRAbort && deltaT < _deltaTAbort) //迭代终止条件
            break;
          
    } //end of L-M method

  }  //end of   if (lastCornerCloudSize > 10 && lastSurfaceCloudSize > 100) 

//坐标转换
//算出了两坨点云间的相对运动，但他们是在这两帧点云的局部坐标系下的，我们需要把它转换到世界坐标系下，因此需要进行转换。
//这部分内容较为简单，直接上代码了：
  Angle rx, ry, rz;
  // 计算旋转角的累计变化量        //求相对于原点的旋转量,垂直方向上1.05倍修正?
  accumulateRotation(_transformSum.rot_x,
                     _transformSum.rot_y,
                     _transformSum.rot_z,
                     -_transform.rot_x,
                     -_transform.rot_y.rad() * 1.05,
                     -_transform.rot_z,
                     rx, ry, rz);

  Vector3 v( _transform.pos.x()        - _imuShiftFromStart.x(),
             _transform.pos.y()        - _imuShiftFromStart.y(),
             _transform.pos.z() * 1.05 - _imuShiftFromStart.z() );
  rotateZXY(v, rz, rx, ry);

  //求相对于原点的平移量
  Vector3 trans = _transformSum.pos - v;

 //根据IMU修正旋转量
  pluginIMURotation(rx, ry, rz,
                    _imuPitchStart, _imuYawStart, _imuRollStart,
                    _imuPitchEnd, _imuYawEnd, _imuRollEnd,
                    rx, ry, rz);
 //得到世界坐标系下的转移矩阵
  _transformSum.rot_x = rx;
  _transformSum.rot_y = ry;
  _transformSum.rot_z = rz;
  _transformSum.pos = trans;

  transformToEnd(_cornerPointsLessSharp);
  transformToEnd(_surfPointsLessFlat);

  _cornerPointsLessSharp.swap(_lastCornerCloud);
  _surfPointsLessFlat.swap(_lastSurfaceCloud);

  lastCornerCloudSize = _lastCornerCloud->points.size();
  lastSurfaceCloudSize = _lastSurfaceCloud->points.size();

  if (lastCornerCloudSize > 10 && lastSurfaceCloudSize > 100) {
    _lastCornerKDTree.setInputCloud(_lastCornerCloud);
    _lastSurfaceKDTree.setInputCloud(_lastSurfaceCloud);
  }

  publishResult();

}  //END OF PROCESS



void LaserOdometry::publishResult()
{
  // publish odometry tranformations       欧拉角转换成四元数
  geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(_transformSum.rot_z.rad(),
                                                                              -_transformSum.rot_x.rad(),
                                                                              -_transformSum.rot_y.rad());
  //publish四元数和平移量
  _laserOdometryMsg.header.stamp = _timeSurfPointsLessFlat;
  _laserOdometryMsg.pose.pose.orientation.x = -geoQuat.y;
  _laserOdometryMsg.pose.pose.orientation.y = -geoQuat.z;
  _laserOdometryMsg.pose.pose.orientation.z = geoQuat.x;
  _laserOdometryMsg.pose.pose.orientation.w = geoQuat.w;
  _laserOdometryMsg.pose.pose.position.x = _transformSum.pos.x();
  _laserOdometryMsg.pose.pose.position.y = _transformSum.pos.y();
  _laserOdometryMsg.pose.pose.position.z = _transformSum.pos.z();
  _pubLaserOdometry.publish(_laserOdometryMsg);

  //广播新的平移旋转之后的坐标系(rviz)
  _laserOdometryTrans.stamp_ = _timeSurfPointsLessFlat;
  _laserOdometryTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
  _laserOdometryTrans.setOrigin(tf::Vector3( _transformSum.pos.x(), _transformSum.pos.y(), _transformSum.pos.z()) );
  _tfBroadcaster.sendTransform(_laserOdometryTrans);


  // publish cloud results according to the input output ratio
  //按照跳帧数publich边沿点，平面点以及全部点给laserMapping(每隔一帧发一次)
  if (_ioRatio < 2 || _frameCount % _ioRatio == 1) 
  {
    ros::Time sweepTime = _timeSurfPointsLessFlat;
    transformToEnd(_laserCloud);  // transform full resolution cloud to sweep end before sending it
        
  //将cornerPointsLessSharp和surfPointLessFlat点也即边沿点和平面点分别发送给laserMapping
    publishCloudMsg(_pubLaserCloudCornerLast, *_lastCornerCloud, sweepTime, "/camera");
    publishCloudMsg(_pubLaserCloudSurfLast, *_lastSurfaceCloud, sweepTime, "/camera");
//_laserCloud 是将上一帧点云中的点相对结束位置去除因匀速运动产生的畸变，效果相当于得到在点云扫描结束位置静止扫描得到的点云
    publishCloudMsg(_pubLaserCloudFullRes, *_laserCloud, sweepTime, "/camera");
  }
}

} // end namespace loam
