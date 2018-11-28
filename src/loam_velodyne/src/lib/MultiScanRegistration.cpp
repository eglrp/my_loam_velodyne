/******************************读前须知*****************************************/
/*imu为x轴向前,y轴向左,z轴向上的右手坐标系，
  velodyne lidar被安装为x轴向前,y轴向左,z轴向上的右手坐标系，
  scanRegistration会把两者通过交换坐标轴，都统一到z轴向前,x轴向左,y轴向上的右手坐标系
  ，这是J. Zhang的论文里面使用的坐标系
  交换后：R = Ry(yaw)*Rx(pitch)*Rz(roll)
*******************************************************************************/

#include "loam_velodyne/MultiScanRegistration.h"
#include "math_utils.h"

#include <pcl_conversions/pcl_conversions.h>


namespace loam {

MultiScanMapper::MultiScanMapper(const float& lowerBound,
                                 const float& upperBound,
                                 const uint16_t& nScanRings)
    : _lowerBound(lowerBound),
      _upperBound(upperBound),
      _nScanRings(nScanRings),
      _factor((nScanRings - 1) / (upperBound - lowerBound))
{

}

void MultiScanMapper::set(const float &lowerBound,
                          const float &upperBound,
                          const uint16_t &nScanRings)
{
  _lowerBound = lowerBound;
  _upperBound = upperBound;
  _nScanRings = nScanRings;
  _factor = (nScanRings - 1) / (upperBound - lowerBound);
}



int MultiScanMapper::getRingForAngle(const float& angle) {
  return int(((angle * 180 / M_PI) - _lowerBound) * _factor + 0.5);
}



MultiScanRegistration::MultiScanRegistration(const MultiScanMapper& scanMapper,
                                             const RegistrationParams& config)
    : ScanRegistration(config),
      _systemDelay(SYSTEM_DELAY),
      _scanMapper(scanMapper)
{

};



bool MultiScanRegistration::setup(ros::NodeHandle& node,ros::NodeHandle& privateNode)
{
  if (!ScanRegistration::setup(node, privateNode)) {
    return false;
  }

  // fetch scan mapping params
  std::string lidarName;

  if (privateNode.getParam("lidar", lidarName))  //读取launch文件中的param
  {
    if (lidarName == "VLP-16") {
      _scanMapper = MultiScanMapper::Velodyne_VLP_16();
    } else if (lidarName == "HDL-32") {
      _scanMapper = MultiScanMapper::Velodyne_HDL_32();
    } else if (lidarName == "HDL-64E") {
      _scanMapper = MultiScanMapper::Velodyne_HDL_64E();
    } else {
      ROS_ERROR("Invalid lidar parameter: %s (only \"VLP-16\", \"HDL-32\" and \"HDL-64E\" are supported)", lidarName.c_str());
      return false;
    }

    ROS_INFO("Set  %s  scan mapper.", lidarName.c_str());
    if (!privateNode.hasParam("scanPeriod"))    //读取launch文件中的param
    {
      _config.scanPeriod = 0.1;
      ROS_INFO("Set scanPeriod: %f", _config.scanPeriod);
    }
  } 
  else 
  {
    float vAngleMin, vAngleMax;
    int nScanRings;

    if (privateNode.getParam("minVerticalAngle", vAngleMin) &&
        privateNode.getParam("maxVerticalAngle", vAngleMax) &&
        privateNode.getParam("nScanRings", nScanRings)) 
      {
        if (vAngleMin >= vAngleMax) 
        {
          ROS_ERROR("Invalid vertical range (min >= max)");
          return false;
        }
      else if (nScanRings < 2) 
        {
          ROS_ERROR("Invalid number of scan rings (n < 2)");
          return false;
        }

        _scanMapper.set(vAngleMin, vAngleMax, nScanRings);
      ROS_INFO("Set linear scan mapper from %g to %g degrees with %d scan rings.", vAngleMin, vAngleMax, nScanRings);
    }
  }

  // subscribe to input cloud topic!!!!!!!!
  //该节点通过订阅器subLaserCloud订阅 /multi_scan_points消息，并由回调函数handleCloudMessage进行处理，订阅器缓存区大小为2；
  _subLaserCloud = node.subscribe<sensor_msgs::PointCloud2>
      ("/multi_scan_points", 2, &MultiScanRegistration::handleCloudMessage, this);
  return true;
}


void MultiScanRegistration::handleCloudMessage(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    if (_systemDelay > 0) //丢弃前20个点云数据
    {
      _systemDelay--;
      return;
    }

    // fetch new input cloud
    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
      //消息转换成pcl数据存放
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);

    process(laserCloudIn, laserCloudMsg->header.stamp);
}

//对接收到的 一帧 ！！！！点云进行预处理，完成分类
//具体分类内容为：一是按照不同线，将点云保存在点云指针中；二是对其进行特征分类。
void MultiScanRegistration::process(const pcl::PointCloud<pcl::PointXYZ>& laserCloudIn, const ros::Time& scanTime)
{
  size_t cloudSize = laserCloudIn.size(); //点云点的数量
  // reset internal buffers and set IMU start state based on current scan time
  reset(scanTime);

  // determine scan start and end orientations
  // 计算起始点和终止点方向
//lidar scan开始点的旋转角,atan2范围[-pi,+pi],计算旋转角时取负号是因为velodyne是顺时针旋转
  float startOri = -std::atan2(laserCloudIn[0].y, laserCloudIn[0].x);
//lidar scan结束点的旋转角，加2*pi使点云旋转周期为2*pi
  float endOri = -std::atan2(laserCloudIn[cloudSize - 1].y,laserCloudIn[cloudSize - 1].x) + 2 * float(M_PI);
//结束方位角与开始方位角差值控制在(PI,3*PI)范围，允许lidar不是一个圆周扫描
//正常情况下在这个范围内：pi < endOri - startOri < 3*pi，异常则修正
  if (endOri - startOri > 3 * M_PI) 
    endOri -= 2 * M_PI;
 else if (endOri - startOri < M_PI) 
    endOri += 2 * M_PI;
  
  //lidar扫描线是否旋转过半
  bool halfPassed = false;
  pcl::PointXYZI point;  //作为临时处理储存点
  std::vector<pcl::PointCloud<pcl::PointXYZI> > laserCloudScans(_scanMapper.getNumberOfScanRings());  //将点云分线

  // extract valid points from input cloud
  //一个点一个点的计算   cloudSize：点云点的数量
  for (int i = 0; i < cloudSize; i++) 
  {
    //坐标轴交换，velodyne lidar的坐标系也转换到z轴向前，x轴向左的右手坐标系
      point.x = laserCloudIn[i].y;  //可以正常建图
      point.y = laserCloudIn[i].z;
      point.z = laserCloudIn[i].x;

      //point.x = laserCloudIn[i].x;   //?????为什么不能建图
      //point.y = laserCloudIn[i].y;
      //point.z = laserCloudIn[i].z;


      // skip NaN and INF valued points  移除空点
      if (!pcl_isfinite(point.x) || !pcl_isfinite(point.y) || !pcl_isfinite(point.z)) 
        continue;
      // skip zero valued points
      if (point.x * point.x + point.y * point.y + point.z * point.z < 0.0001) 
        continue;
      

      // calculate vertical point angle and scan ID
 //计算点的仰角(根据lidar文档垂直角计算公式),根据仰角排列激光线号，velodyne每两个scan之间间隔2度
      float angle = std::atan(point.y / std::sqrt(point.x * point.x + point.z * point.z));
      int scanID = _scanMapper.getRingForAngle(angle);   //确定是第几线
      if (scanID >= _scanMapper.getNumberOfScanRings() || scanID < 0 )  
        continue;
      
      // calculate horizontal point angle
    //该点的旋转角
      float ori = -std::atan2(point.x, point.z);

      //根据扫描线是否旋转过半选择与起始位置还是终止位置进行差值计算，从而进行补偿
      if (!halfPassed) 
      {
        //确保-pi/2 < ori - startOri < 3*pi/2
        if (ori < startOri - M_PI / 2) 
          ori += 2 * M_PI;
        else if (ori > startOri + M_PI * 3 / 2) 
          ori -= 2 * M_PI;
        
        if (ori - startOri > M_PI) 
          halfPassed = true;
      } 
      else 
      {
        ori += 2 * M_PI;
        if (ori < endOri - M_PI * 3 / 2) 
          ori += 2 * M_PI;
        else if (ori > endOri + M_PI / 2) 
          ori -= 2 * M_PI;
      }

      // calculate relative scan time based on point orientation
      //-0.5 < relTime < 1.5（点旋转的角度与整个周期旋转角度的比率, 即点云中点的相对时间）
      float relTime = _config.scanPeriod * (ori - startOri) / (endOri - startOri);
          //点强度=线号+点相对时间（即一个整数+一个小数，整数部分是线号，小数部分是该点的相对时间）,
          //匀速扫描：根据当前扫描的角度和扫描周期计算相对扫描起始位置的时间
      point.intensity = scanID + relTime; // 点的intensity: 整数部分为scanID， 小数部分为扫描时间

      // project point to the start of the sweep using corresponding IMU data
      if (hasIMUData()) {//如果收到IMU数据,使用IMU矫正点云畸变
        setIMUTransformFor(relTime);
        transformToStartIMU(point);
      }

      laserCloudScans[scanID].push_back(point);  //将每个补偿矫正的点放入对应线号的容器
  }

  // construct sorted full resolution cloud
  cloudSize = 0;
  for (int i = 0; i < _scanMapper.getNumberOfScanRings(); i++) 
  { //将所有的点按照线号从小到大放入一个容器
    _laserCloud += laserCloudScans[i];   //_laserCloud  full resolution input cloud
    IndexRange range(cloudSize, 0);  //typedef std::pair<size_t, size_t> IndexRange;
    cloudSize += laserCloudScans[i].size();
    range.second = cloudSize > 0 ? cloudSize - 1 : 0;
    _scanIndices.push_back(range);
  }

  // extract features  继承于ScanRegistration
  extractFeatures();
  // publish result
  publishResult();
}

} // end namespace loam
