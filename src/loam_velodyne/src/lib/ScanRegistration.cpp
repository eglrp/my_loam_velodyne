
//对点云和IMU数据进行预处理，用于特征点的配准。,提取特征点，并且发布出去
//根据点的曲率c来将点划分为不同的类别（边/面特征或不是特征）
#include "loam_velodyne/ScanRegistration.h"
#include "math_utils.h"

#include <pcl/filters/voxel_grid.h>
#include <tf/transform_datatypes.h>


namespace loam {


RegistrationParams::RegistrationParams(const float& scanPeriod_,
                                       const int& imuHistorySize_,
                                       const int& nFeatureRegions_,
                                       const int& curvatureRegion_,
                                       const int& maxCornerSharp_,
                                       const int& maxSurfaceFlat_,
                                       const float& lessFlatFilterSize_,
                                       const float& surfaceCurvatureThreshold_)
    : scanPeriod(scanPeriod_),
      imuHistorySize(imuHistorySize_),
      nFeatureRegions(nFeatureRegions_),
      curvatureRegion(curvatureRegion_),
      maxCornerSharp(maxCornerSharp_),
      maxCornerLessSharp(10 * maxCornerSharp_),
      maxSurfaceFlat(maxSurfaceFlat_),
      lessFlatFilterSize(lessFlatFilterSize_),
      surfaceCurvatureThreshold(surfaceCurvatureThreshold_)
{

};


/*
Params
*/
bool RegistrationParams::parseParams(const ros::NodeHandle& nh) 
{
  bool success = true;
  int iParam = 0;
  float fParam = 0;

  if (nh.getParam("scanPeriod", fParam)) {
    if (fParam <= 0) {
      ROS_ERROR("Invalid scanPeriod parameter: %f (expected > 0)", fParam);
      success = false;
    } else {
      scanPeriod = fParam;
      ROS_INFO("Set scanPeriod: %g", fParam);
    }
  }

  if (nh.getParam("imuHistorySize", iParam)) {
    if (iParam < 1) {
      ROS_ERROR("Invalid imuHistorySize parameter: %d (expected >= 1)", iParam);
      success = false;
    } else {
      imuHistorySize = iParam;
      ROS_INFO("Set imuHistorySize: %d", iParam);
    }
  }

  if (nh.getParam("featureRegions", iParam)) {
    if (iParam < 1) {
      ROS_ERROR("Invalid featureRegions parameter: %d (expected >= 1)", iParam);
      success = false;
    } else {
      nFeatureRegions = iParam;
      ROS_INFO("Set nFeatureRegions: %d", iParam);
    }
  }

  if (nh.getParam("curvatureRegion", iParam)) {
    if (iParam < 1) {
      ROS_ERROR("Invalid curvatureRegion parameter: %d (expected >= 1)", iParam);
      success = false;
    } else {
      curvatureRegion = iParam;
      ROS_INFO("Set curvatureRegion: +/- %d", iParam);
    }
  }

  if (nh.getParam("maxCornerSharp", iParam)) {
    if (iParam < 1) {
      ROS_ERROR("Invalid maxCornerSharp parameter: %d (expected >= 1)", iParam);
      success = false;
    } else {
      maxCornerSharp = iParam;
      maxCornerLessSharp = 10 * iParam;
      ROS_INFO("Set maxCornerSharp / less sharp: %d / %d", iParam, maxCornerLessSharp);
    }
  }

  if (nh.getParam("maxCornerLessSharp", iParam)) {
    if (iParam < maxCornerSharp) {
      ROS_ERROR("Invalid maxCornerLessSharp parameter: %d (expected >= %d)", iParam, maxCornerSharp);
      success = false;
    } else {
      maxCornerLessSharp = iParam;
      ROS_INFO("Set maxCornerLessSharp: %d", iParam);
    }
  }

  if (nh.getParam("maxSurfaceFlat", iParam)) {
    if (iParam < 1) {
      ROS_ERROR("Invalid maxSurfaceFlat parameter: %d (expected >= 1)", iParam);
      success = false;
    } else {
      maxSurfaceFlat = iParam;
      ROS_INFO("Set maxSurfaceFlat: %d", iParam);
    }
  }

  if (nh.getParam("surfaceCurvatureThreshold", fParam)) {
    if (fParam < 0.001) {
      ROS_ERROR("Invalid surfaceCurvatureThreshold parameter: %f (expected >= 0.001)", fParam);
      success = false;
    } else {
      surfaceCurvatureThreshold = fParam;
      ROS_INFO("Set surfaceCurvatureThreshold: %g", fParam);
    }
  }

  if (nh.getParam("lessFlatFilterSize", fParam)) {
    if (fParam < 0.001) {
      ROS_ERROR("Invalid lessFlatFilterSize parameter: %f (expected >= 0.001)", fParam);
      success = false;
    } else {
      lessFlatFilterSize = fParam;
      ROS_INFO("Set lessFlatFilterSize: %g", fParam);
    }
  }

  return success;
}

/*
参数初始化列表
*/
ScanRegistration::ScanRegistration(const RegistrationParams& config)
      : _config(config),
        _sweepStart(),
        _scanTime(),
        _imuStart(),
        _imuCur(),
        _imuIdx(0),
        _imuHistory(_config.imuHistorySize),
        _laserCloud(),
        _cornerPointsSharp(),
        _cornerPointsLessSharp(),
        _surfacePointsFlat(),
        _surfacePointsLessFlat(),
        _imuTrans(4, 1),
        _regionCurvature(),
        _regionLabel(),
        _regionSortIndices(),
        _scanNeighborPicked()
{

}
/*
订阅、发布话题的设置
*/
bool ScanRegistration::setup(ros::NodeHandle& node, ros::NodeHandle& privateNode)
{
  if (!_config.parseParams(privateNode)) {
    return false;
  }
  _imuHistory.ensureCapacity(_config.imuHistorySize);

  // subscribe to IMU topic
  //订阅器_subImu订阅 /imu/data消息，由回调函数handleIMUMessage处理，缓存区大小为50。
  _subImu = node.subscribe<sensor_msgs::Imu>("/imu/data", 50, &ScanRegistration::handleIMUMessage, this);


  // advertise scan registration topics
  /*此外，该节点通过发布器_pubLaserCloud发布以不同线分类的点云的指针/velodyne_cloud_2，缓存区大小为2；
  发布器_pubCornerPointsSharp和_pubCornerPointsLessSharp发布特征角点 /laser_cloud_sharp和/laser_cloud_less_sharp，缓存区大小为2；
  发布器_pubSurfPointFlat和_pubSurfPointLessFlat发布特征角点 /laser_cloud_flat和/laser_cloud_less_flat，缓存区大小为2；
  发布器_pubImuTrans发布IMU信息 /imu_trans，缓存区大小为5。下面从两个回调函数分别介绍这个节点的代码。*/
  _pubLaserCloud = node.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 2); // 发布分成多线registrated  点云 
  _pubCornerPointsSharp = node.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 2);
  _pubCornerPointsLessSharp = node.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 2); 
  _pubSurfPointsFlat = node.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 2);
  _pubSurfPointsLessFlat = node.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 2);
  _pubImuTrans = node.advertise<sensor_msgs::PointCloud2>("/imu_trans", 5);

  return true;
}

/*
处理imu初始数据
接收imu消息，imu坐标系为x轴向前，y轴向右，z轴向上的右手坐标系
*/
void ScanRegistration::handleIMUMessage(const sensor_msgs::Imu::ConstPtr& imuIn)
{
  double roll, pitch, yaw;
  tf::Quaternion orientation;
 //convert Quaternion msg to Quaternion
  tf::quaternionMsgToTF(imuIn->orientation, orientation);
 //This will get the roll pitch and yaw from the matrix about fixed axes X, Y, Z respectively. That's R = Rz(yaw)*Ry(pitch)*Rx(roll).
 //Here roll pitch yaw is in the global frame
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

  //减去重力的影响,求出xyz方向的加速度实际值，并进行坐标轴交换，统一到z轴向前,x轴向左的右手坐标系, 
  //交换过后RPY对应fixed axes ZXY(RPY---ZXY)。Now R = Ry(yaw)*Rx(pitch)*Rz(roll).
  Vector3 acc;
  acc.x() = float(imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81);
  acc.y() = float(imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81);
  acc.z() = float(imuIn->linear_acceleration.x + sin(pitch) * 9.81);

  IMUState newState;
  newState.stamp = imuIn->header.stamp;
  newState.roll = roll;
  newState.pitch = pitch;
  newState.yaw = yaw;
  newState.acceleration = acc;

  if (_imuHistory.size() > 0) 
  {
    // accumulate IMU position and velocity over time
    rotateZXY(acc, newState.roll, newState.pitch, newState.yaw);

    const IMUState& prevState = _imuHistory.last();
    float timeDiff = float((newState.stamp - prevState.stamp).toSec());
    newState.position = prevState.position
                        + (prevState.velocity * timeDiff)
                        + (0.5 * acc * timeDiff * timeDiff);
    newState.velocity = prevState.velocity
                        + acc * timeDiff;
  }

  _imuHistory.push(newState);
}


void ScanRegistration::reset(const ros::Time& scanTime, const bool& newSweep)
{
  _scanTime = scanTime;

  // re-initialize IMU start index and state
  _imuIdx = 0;
  if (hasIMUData()) {
    interpolateIMUStateFor(0, _imuStart);
  }

  // clear internal cloud buffers at the beginning of a sweep
  if (newSweep) 
  {
    _sweepStart = scanTime;

    // clear cloud buffers
    _laserCloud.clear();
    _cornerPointsSharp.clear();
    _cornerPointsLessSharp.clear();
    _surfacePointsFlat.clear();
    _surfacePointsLessFlat.clear();

    // clear scan indices vector
    _scanIndices.clear();
  }
}


void ScanRegistration::setIMUTransformFor(const float& relTime)
{
  interpolateIMUStateFor(relTime, _imuCur);

  float relSweepTime = (_scanTime - _sweepStart).toSec() + relTime;
  _imuPositionShift = _imuCur.position - _imuStart.position - _imuStart.velocity * relSweepTime;
}


void ScanRegistration::transformToStartIMU(pcl::PointXYZI& point)
{
  // rotate point to global IMU system
  rotateZXY(point, _imuCur.roll, _imuCur.pitch, _imuCur.yaw);

  // add global IMU position shift
  point.x += _imuPositionShift.x();
  point.y += _imuPositionShift.y();
  point.z += _imuPositionShift.z();

  // rotate point back to local IMU system relative to the start IMU state
  rotateYXZ(point, -_imuStart.yaw, -_imuStart.pitch, -_imuStart.roll);
}

void ScanRegistration::interpolateIMUStateFor(const float &relTime,IMUState &outputState)
{
  double timeDiff = (_scanTime - _imuHistory[_imuIdx].stamp).toSec() + relTime;
  while (_imuIdx < _imuHistory.size() - 1 && timeDiff > 0) {
    _imuIdx++;
    timeDiff = (_scanTime - _imuHistory[_imuIdx].stamp).toSec() + relTime;
  }

  if (_imuIdx == 0 || timeDiff > 0) {
    outputState = _imuHistory[_imuIdx];
  } 
  else 
  {
    float ratio = -timeDiff / (_imuHistory[_imuIdx].stamp - _imuHistory[_imuIdx - 1].stamp).toSec();
    IMUState::interpolate(_imuHistory[_imuIdx], _imuHistory[_imuIdx - 1], ratio, outputState);
  }
}

/*
  计算每一线 平均分开的区域里点的曲率
*/
void ScanRegistration::setRegionBuffersFor(const size_t& startIdx, const size_t& endIdx)
{
  // resize buffers
  size_t regionSize = endIdx - startIdx + 1;
  _regionCurvature.resize(regionSize);
  _regionSortIndices.resize(regionSize);
  _regionLabel.assign(regionSize, SURFACE_LESS_FLAT);

  // calculate point curvatures and reset sort indices
  float pointWeight = -2 * _config.curvatureRegion;  //-10

  for (size_t i = startIdx, regionIdx = 0; i <= endIdx; i++, regionIdx++) 
  {
    float diffX = pointWeight * _laserCloud[i].x;
    float diffY = pointWeight * _laserCloud[i].y;
    float diffZ = pointWeight * _laserCloud[i].z;

    for (int j = 1; j <= _config.curvatureRegion; j++) //使用每个点的前后五个点计算曲率
    {
      diffX += _laserCloud[i + j].x + _laserCloud[i - j].x;
      diffY += _laserCloud[i + j].y + _laserCloud[i - j].y;
      diffZ += _laserCloud[i + j].z + _laserCloud[i - j].z;
    }
    //曲率计算 _regionCurvature是存储每个点的曲率 由之前的laserCloud+=……，
    //可以知道的是这个时候的点云是按照线的数据方式存储的，线的数量自己定义。
    _regionCurvature[regionIdx] = diffX * diffX + diffY * diffY + diffZ * diffZ;
   //记录曲率点的索引
    _regionSortIndices[regionIdx] = i;
  }

  // sort point curvatures
  for (size_t i = 1; i < regionSize; i++) 
  {
    for (size_t j = i; j >= 1; j--) 
    {
      if (_regionCurvature[_regionSortIndices[j] - startIdx] < _regionCurvature[_regionSortIndices[j - 1] - startIdx]) 
        std::swap(_regionSortIndices[j], _regionSortIndices[j - 1]);
    }
  }
}

/*
  筛选每一层的点
*/
void ScanRegistration::setScanBuffersFor(const size_t& startIdx,const size_t& endIdx)
{
  // resize buffers
  size_t scanSize = endIdx - startIdx + 1;
  _scanNeighborPicked.assign(scanSize, 0);

  // mark unreliable points as picked
  //挑选点，排除容易被斜面挡住的点以及离群点，有些点容易被斜面挡住，而离群点可能出现带有偶然性，这些情况都可能导致前后两次扫描不能被同时看到
  for (size_t i = startIdx + _config.curvatureRegion; i < endIdx - _config.curvatureRegion; i++) 
  {                                                       //与后一个点差值，所以减6
    const pcl::PointXYZI& previousPoint = (_laserCloud[i - 1]);
    const pcl::PointXYZI& point = (_laserCloud[i]);
    const pcl::PointXYZI& nextPoint = (_laserCloud[i + 1]);
    //计算有效曲率点与后一个点之间的距离平方和
    float diffNext = calcSquaredDiff(nextPoint, point);

    if (diffNext > 0.1) //前提:两个点之间距离要大于0.1
    {
      float depth1 = calcPointDistance(point);        //点的深度
      float depth2 = calcPointDistance(nextPoint);        //后一个点的深度

      if (depth1 > depth2)         //按照两点的深度的比例，将深度较大的点拉回后计算距离
      {
        float weighted_distance = std::sqrt( calcSquaredDiff(nextPoint, point, depth2/depth1) ) / depth2;
    //边长比也即是弧度值，若小于0.1，说明夹角比较小，斜面比较陡峭,点深度变化比较剧烈,点处在近似与激光束平行的斜面上
        if (weighted_distance < 0.1) //排除容易被斜面挡住的点
        { //该点及前面五个点（大致都在斜面上）全部置为筛选过
          std::fill_n(&_scanNeighborPicked[i - startIdx - _config.curvatureRegion], _config.curvatureRegion + 1, 1);

          continue;
        }
      } 
      else 
      {
        float weighted_distance = std::sqrt(calcSquaredDiff(point, nextPoint, depth1 / depth2)) / depth1;

        if (weighted_distance < 0.1) {
          std::fill_n(&_scanNeighborPicked[i - startIdx + 1], _config.curvatureRegion + 1, 1);
        }
      }
    }

    float diffPrevious = calcSquaredDiff(point, previousPoint);
    float dis = calcSquaredPointDistance(point);

    //与前后点的平方和都大于深度平方和的万分之二，这些点视为离群点，包括陡斜面上的点，强烈凸凹点和空旷区域中的某些点，置为筛选过，弃用
    if (diffNext > 0.0002 * dis && diffPrevious > 0.0002 * dis) 
      _scanNeighborPicked[i - startIdx] = 1;
    
  }
}



void ScanRegistration::markAsPicked(const size_t& cloudIdx,const size_t& scanIdx)
{
  _scanNeighborPicked[scanIdx] = 1;

  for (int i = 1; i <= _config.curvatureRegion; i++) 
  {
    if (calcSquaredDiff(_laserCloud[cloudIdx + i], _laserCloud[cloudIdx + i - 1]) > 0.05) 
      break;

    _scanNeighborPicked[scanIdx + i] = 1;
  }

  for (int i = 1; i <= _config.curvatureRegion; i++) 
  {
    if (calcSquaredDiff(_laserCloud[cloudIdx - i], _laserCloud[cloudIdx - i + 1]) > 0.05) 
      break;
    
    _scanNeighborPicked[scanIdx - i] = 1;
  }
}

void ScanRegistration::extractFeatures(const uint16_t& beginIdx)
{
  // extract features from individual scans
   size_t nScans = _scanIndices.size();  //点云数据的线数
   //ROS_INFO("!!!!!!!!!!!!!!!!!!!!!: %d", nScans);

  for (size_t i = beginIdx; i < nScans; i++) //每一线数据分别处理
  {
      pcl::PointCloud<pcl::PointXYZI>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<pcl::PointXYZI>);
      size_t scanStartIdx = _scanIndices[i].first;
      size_t scanEndIdx = _scanIndices[i].second;

      // skip empty scans                            =5
      if (scanEndIdx <= scanStartIdx + 2 * _config.curvatureRegion) //单层数据   最后一个点没有比起始点大10
        continue;      

      // Quick&Dirty fix for relative point time calculation without IMU data
      /*float scanSize = scanEndIdx - scanStartIdx + 1;
      for (int j = scanStartIdx; j <= scanEndIdx; j++) {
        _laserCloud[j].intensity = i + _scanPeriod * (j - scanStartIdx) / scanSize;
      }*/

      // reset scan buffers 
      setScanBuffersFor(scanStartIdx, scanEndIdx);  //单层数据  筛选每一层符合条件的点

      // extract features from equally sized scan regions  =6
      for (int j = 0; j < _config.nFeatureRegions; j++) 
      {
        //计算特征区域   将每一线划分为等间距的6段分别处理，在每一段升序排列。 变量说明sp 每一段的startPoint；ep endPoint.
        size_t sp = ((scanStartIdx + _config.curvatureRegion) * (_config.nFeatureRegions - j)
                               + (scanEndIdx - _config.curvatureRegion) * j) / _config.nFeatureRegions; //sp startPoint

        size_t ep = ((scanStartIdx + _config.curvatureRegion) * (_config.nFeatureRegions - 1 - j)
                         + (scanEndIdx - _config.curvatureRegion) * (j + 1)) / _config.nFeatureRegions - 1;//ep endPoint.
        // skip empty regions
        if (ep <= sp) 
          continue;
        size_t regionSize = ep - sp + 1;

        // reset region buffers
        setRegionBuffersFor(sp, ep);  // 计算每一线 平均分开的区域里点的曲率

        // extract corner features
        int largestPickedNum = 0;
        for (size_t k = regionSize; k > 0 && largestPickedNum < _config.maxCornerLessSharp;) 
        {
            size_t idx = _regionSortIndices[--k];
            size_t scanIdx = idx - scanStartIdx;
            size_t regionIdx = idx - sp;
 
             if (_scanNeighborPicked[scanIdx] == 0 && _regionCurvature[regionIdx] > _config.surfaceCurvatureThreshold) 
              {
                largestPickedNum++;
                if (largestPickedNum <= _config.maxCornerSharp) 
                {
                  _regionLabel[regionIdx] = CORNER_SHARP;
                  _cornerPointsSharp.push_back(_laserCloud[idx]);  //_cornerPointsSharp
                } 
                else 
                {
                  _regionLabel[regionIdx] = CORNER_LESS_SHARP;
                }

                _cornerPointsLessSharp.push_back(_laserCloud[idx]);  //_cornerPointsLessSharp
              markAsPicked(idx, scanIdx);
            }
        }

        // extract flat surface features
        int smallestPickedNum = 0;
        for (int k = 0; k < regionSize && smallestPickedNum < _config.maxSurfaceFlat; k++) 
        {
          size_t idx = _regionSortIndices[k];
          size_t scanIdx = idx - scanStartIdx;
          size_t regionIdx = idx - sp;

          if (_scanNeighborPicked[scanIdx] == 0 &&_regionCurvature[regionIdx] < _config.surfaceCurvatureThreshold) 
          {
            smallestPickedNum++;
            _regionLabel[regionIdx] = SURFACE_FLAT;
            _surfacePointsFlat.push_back(_laserCloud[idx]);

            markAsPicked(idx, scanIdx);
          }
        }
        // extract less flat surface features
        for (int k = 0; k < regionSize; k++) 
        {
          if (_regionLabel[k] <= SURFACE_LESS_FLAT) 
            surfPointsLessFlatScan->push_back(_laserCloud[sp + k]);          
        }
      }  // end of extract features from equally sized scan regions  =6

      // down size less flat surface point cloud of current scan
      pcl::PointCloud<pcl::PointXYZI> surfPointsLessFlatScanDS;
      pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
      downSizeFilter.setInputCloud(surfPointsLessFlatScan);
      //修改lessFlatFilterSize 影响建图效果
      downSizeFilter.setLeafSize(_config.lessFlatFilterSize, _config.lessFlatFilterSize, _config.lessFlatFilterSize);
      downSizeFilter.filter(surfPointsLessFlatScanDS);

      _surfacePointsLessFlat += surfPointsLessFlatScanDS;
  }

}

void ScanRegistration::publishResult()
{
  // publish full resolution and feature point clouds
  publishCloudMsg(_pubLaserCloud,            _laserCloud,            _sweepStart, "/camera");
  publishCloudMsg(_pubCornerPointsSharp,     _cornerPointsSharp,     _sweepStart, "/camera");
  publishCloudMsg(_pubCornerPointsLessSharp, _cornerPointsLessSharp, _sweepStart, "/camera");
  publishCloudMsg(_pubSurfPointsFlat,        _surfacePointsFlat,     _sweepStart, "/camera");
  publishCloudMsg(_pubSurfPointsLessFlat,    _surfacePointsLessFlat, _sweepStart, "/camera");


  // publish corresponding IMU transformation information
  _imuTrans[0].x = _imuStart.pitch.rad();
  _imuTrans[0].y = _imuStart.yaw.rad();
  _imuTrans[0].z = _imuStart.roll.rad();

  _imuTrans[1].x = _imuCur.pitch.rad();
  _imuTrans[1].y = _imuCur.yaw.rad();
  _imuTrans[1].z = _imuCur.roll.rad();

  Vector3 imuShiftFromStart = _imuPositionShift;
  rotateYXZ(imuShiftFromStart, -_imuStart.yaw, -_imuStart.pitch, -_imuStart.roll);

  _imuTrans[2].x = imuShiftFromStart.x();
  _imuTrans[2].y = imuShiftFromStart.y();
  _imuTrans[2].z = imuShiftFromStart.z();

  Vector3 imuVelocityFromStart = _imuCur.velocity - _imuStart.velocity;
  rotateYXZ(imuVelocityFromStart, -_imuStart.yaw, -_imuStart.pitch, -_imuStart.roll);

  _imuTrans[3].x = imuVelocityFromStart.x();
  _imuTrans[3].y = imuVelocityFromStart.y();
  _imuTrans[3].z = imuVelocityFromStart.z();

  publishCloudMsg(_pubImuTrans, _imuTrans, _sweepStart, "/camera");  ///imu_trans PointCloud2
}

} // end namespace loam
