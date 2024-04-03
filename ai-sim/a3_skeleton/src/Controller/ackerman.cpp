#include "ackerman.h"
#include "Processing/processing.h"
#include "geometry_msgs/Pose.h"
#include "nav_msgs/Odometry.h"
#include "ros/node_handle.h"
#include "sensor_msgs/LaserScan.h"
#include "std_msgs/ColorRGBA.h"
#include "std_msgs/Float64.h"
#include "tf/LinearMath/Quaternion.h"
#include "tf/transform_datatypes.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include <chrono>
#include <cmath>
#include <iterator>
#include <mutex>
#include <pfms_types.h>
#include <sys/stat.h>
#include <thread>

Ackerman::Ackerman() : Ackerman(ros::NodeHandle(), false) {}

Ackerman::Ackerman(ros::NodeHandle nh, bool advanced) : Controller(nh) {
  advanced_ = advanced;
  nh_ = nh;
  initTopics();
  platformType_ = pfms::PlatformType::ACKERMAN;
  std::lock_guard<std::mutex> lk(mtx_);
  platformRunning_ = true;
  status_ = pfms::PlatformStatus::RUNNING;
  runningThread_ = new std::thread(&Ackerman::run, this);
  visConeThread_ = new std::thread(&Ackerman::visualiseCones, this);
  visRoadThread_ = new std::thread(&Ackerman::visualiseRoadCentres, this);
}

Ackerman::~Ackerman() {
  platformRunning_ = false;
  visConeThread_->join();
  visRoadThread_->join();
};

bool Ackerman::initTopics() {
  odoSub_ = nh_.subscribe("/ugv_odom", 1000, &Ackerman::odoCallback, this);
  laserSub_ = nh_.subscribe("/orange/laser/scan", 1000, &Ackerman::laserCallback, this);
  goalSub_ = nh_.subscribe("/orange/goals", 1000, &Ackerman::goalsCallback, this);

  visPub_ = nh_.advertise<visualization_msgs::MarkerArray>("/visualization_marker", 5, false);
  brakePub_ = nh_.advertise<std_msgs::Float64>("/orange/brake_cmd", 5, false);
  steeringPub_ = nh_.advertise<std_msgs::Float64>("/orange/steering_cmd", 5, false);
  throttlePub_ = nh_.advertise<std_msgs::Float64>("/orange/throttle_cmd", 5, false);
  conesPub_ = nh_.advertise<geometry_msgs::PoseArray>("/orange/cones", 5, false);

  goNoGoService_ = nh_.advertiseService("/orange/mission", &Ackerman::request, this);

  return true;
}

void Ackerman::run() {
  nav_msgs::Odometry estimatedGoalPose;
  double distToGoal;
  double timeToGoal;

  while (platformRunning_) {
    while (getStatus() != pfms::PlatformStatus::IDLE and (goalSet_ or advanced_) and !roadCentres_.data.poses.empty()) {
      updateGoals();
      double lastDist = distanceToGoal();
      checkOriginToDestination(getOdometry(), currentGoal_, distToGoal, timeToGoal, estimatedGoalPose);
      {
        std::lock_guard<std::mutex> lk(stats_.mtx);
        stats_.distance_to_goal = distToGoal;
        stats_.time_to_goal = timeToGoal;
      }
      if (goalReached()) {
        break;
      }

      if (!Processing::checkHeadingIsClear(getOdometry(), getScan(), 8)) {
        platformRunning_ = false;
        status_ = pfms::PlatformStatus::IDLE;
        break;
      }

      double optimalBraking = 0;

      // Calculate desired braking force based on distance to goal and current velocity
      if (distToGoal < wheelBase_) {
        odo_.mtx.lock();
        double vel = sqrt(pow(odo_.data.twist.twist.linear.x, 2) + pow(odo_.data.twist.twist.linear.y, 2)); // get current velocity
        odo_.mtx.unlock();
        optimalBraking = maxBrakeTorque_ * 0.44 * pow(vel - 1.5, 1.0 / 3.0) + (maxBrakeTorque_ * 0.44); // do brake calculation
        optimalBraking = optimalBraking > maxBrakeTorque_ ? maxBrakeTorque_ : optimalBraking;           // check for over-braking
        optimalBraking = optimalBraking < 0 ? 0 : optimalBraking;                                       // check for under-braking
        if (isnan(optimalBraking))                                                                      // check for math errors (NaN)
          optimalBraking = 0;
      }

      sendCommand(throttle_, steerAngle_ * steeringRatio_, optimalBraking);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      {
        std::lock_guard<std::mutex> lk(stats_.mtx);
        stats_.time_travelled += 0.010;
        if (distToGoal > 0) {
          stats_.distance_travelled += (lastDist - abs(stats_.distance_to_goal));
          lastDist = stats_.distance_to_goal;
        }
      }
    }

    sendCommand(0, 0, 8000);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (status_ != pfms::PlatformStatus::IDLE) {
      updateGoals();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

bool Ackerman::checkOriginToDestination(nav_msgs::Odometry origin, const geometry_msgs::Pose goal, double &distance, double &time, nav_msgs::Odometry &estimatedGoalPose) {
  double crow = atan2(goal.position.y - origin.pose.pose.position.y, goal.position.x - origin.pose.pose.position.x);
  double yaw = tf::getYaw(origin.pose.pose.orientation);
  double alpha = crow - yaw;

  double ld = sqrt(pow(goal.position.y - origin.pose.pose.position.y, 2.0) + pow(goal.position.x - origin.pose.pose.position.x, 2.0));
  steerAngle_ = atan((2 * wheelBase_ * sin(alpha)) / ld);
  double radius = wheelBase_ / tan(steerAngle_);

  if (abs(steerAngle_) > maxSteerAngle_) {
    estimatedGoalPose = origin;
    distance = INFINITY;
    time = INFINITY;
    return false;
  }

  double temp = alpha > M_PI ? alpha - 2 * M_PI : alpha;
  temp = temp < -M_PI ? temp + 2 * M_PI : temp;

  distance = abs(radius * 2 * temp);
  time = distance / (throttle_ * 29.1);

  // If not a max speed, increase estimated time
  if (std::hypot(origin.twist.twist.linear.x, origin.twist.twist.linear.y) < throttle_ * 29.1)
    time *= 3;

  if (sqrt(pow(goal.position.x - origin.pose.pose.position.x, 2.0) + pow(goal.position.y - origin.pose.pose.position.y, 2.0)) < tolerance_) {
    estimatedGoalPose = origin;
    return true;
  }

  double estimatedYaw = yaw + (2 * alpha);
  estimatedGoalPose = origin;
  estimatedGoalPose.pose.pose.position.x = goal.position.x;
  estimatedGoalPose.pose.pose.position.y = goal.position.y;
  estimatedGoalPose.twist.twist.linear.x = 0;
  estimatedGoalPose.twist.twist.linear.y = 0;

  if (estimatedYaw > M_PI)
    estimatedYaw -= 2 * M_PI;
  if (estimatedYaw < -M_PI)
    estimatedYaw += 2 * M_PI;
  estimatedGoalPose.pose.pose.orientation.w = cos(estimatedYaw / 2);
  estimatedGoalPose.pose.pose.orientation.x = 0;
  estimatedGoalPose.pose.pose.orientation.y = 0;
  estimatedGoalPose.pose.pose.orientation.z = sin(estimatedYaw / 2);

  return true;
};

bool Ackerman::sendCommand(double throttle, double steering, double brake) {
  std_msgs::Float64 msg;
  msg.data = throttle;
  throttlePub_.publish(msg);
  msg.data = steering;
  steeringPub_.publish(msg);
  msg.data = brake;
  brakePub_.publish(msg);
  return false;
};

void Ackerman::odoCallback(const nav_msgs::Odometry::ConstPtr &msg) { setOdometry(*msg); }

void Ackerman::goalsCallback(const geometry_msgs::PoseArrayConstPtr &pose_array_msg) { addGoals(pose_array_msg->poses); };

void Ackerman::laserCallback(const sensor_msgs::LaserScan::ConstPtr &msg) {
  std::lock_guard<std::mutex> lk(laserData_.mtx);
  laserData_.data = *msg;
};

bool Ackerman::request(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res) {
  if (req.data) {
    status_ = pfms::PlatformStatus::RUNNING;
  } else {
    status_ = pfms::PlatformStatus::IDLE;
  }
  nav_msgs::Odometry currentLocation = getOdometry();
  double yaw = tf::getYaw(currentLocation.pose.pose.orientation);
  currentLocation.pose.pose.position.x += laserOffset_ * cos(yaw);
  currentLocation.pose.pose.position.y += laserOffset_ * sin(yaw);
  res.success = Processing::detectCones(currentLocation, getScan(), cones_.data.poses, miscObjects_);
  res.success = Processing::findConePairCentres(getOdometry(), cones_.data.poses, roadCentres_.data.poses);

  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  switch (status_) {
  case pfms::PlatformStatus::RUNNING:
    res.message = "Running";
    break;
  case pfms::PlatformStatus::IDLE:
    res.message = "Idle";
  default:
    break;
  }

  return true;
};

bool Ackerman::updateGoals() {
  std::lock_guard<std::mutex> lk(goals_.mtx);
  if (goals_.data.empty() and !advanced_) {
    goalSet_ = false;
    return false;
  }

  if (advanced_ and roadCentres_.data.poses.empty()) {
    goalSet_ = false;
    return false;
  }

  if (goalSet_) {
    if (goalReached()) {
      goalSet_ = false;
      for (auto g = goals_.data.begin(); g != goals_.data.end(); g++) {
        if (g->position.x == currentGoal_.position.x and g->position.y == currentGoal_.position.y) {
          goals_.data.erase(g);
          return false;
        }
      }
    }
  } else {

    double dist;
    double minDist = INFINITY;
    double temp;
    geometry_msgs::Pose target;
    nav_msgs::Odometry estimated;
    bool found = false;
    for (int i = 0; i < goals_.data.size(); i++) {
      bool res = checkOriginToDestination(getOdometry(), goals_.data.at(i), dist, temp, estimated);
      if (dist < minDist and abs(dist) > tolerance_) {
        found = true;
        minDist = dist;
        target = goals_.data.at(i);
      }
    }
    checkOriginToDestination(getOdometry(), target, dist, temp, estimated);
    double tempSteer = steerAngle_;

    std::vector<geometry_msgs::Pose> newCones;
    std::vector<geometry_msgs::Pose> newCentres;
    nav_msgs::Odometry currentLocation = getOdometry();
    double yaw = tf::getYaw(currentLocation.pose.pose.orientation);
    currentLocation.pose.pose.position.x += laserOffset_ * cos(yaw);
    currentLocation.pose.pose.position.y += laserOffset_ * sin(yaw);

    Processing::detectCones(currentLocation, getScan(), newCones, miscObjects_);
    std::sort(newCones.begin(), newCones.end(), [&](geometry_msgs::Pose p1, geometry_msgs::Pose p2) {
      double dist1;
      double dist2;
      double time;
      nav_msgs::Odometry estimated;
      checkOriginToDestination(getOdometry(), p1, dist1, time, estimated);
      checkOriginToDestination(getOdometry(), p2, dist2, time, estimated);
      return dist1 < dist2;
    });

    bool res = Processing::findConePairCentres(getOdometry(), newCones, newCentres);
    for (auto p = newCentres.begin(); p != newCentres.end();) {
      bool failed = false;
      for (auto c : cones_.data.poses) {
        if (Processing::poseCheckCoincident(*p, c, 3)) {
          failed = true;
          newCentres.erase(p);
          break;
        }
      }
      if (!failed)
        p++;
    }
    Processing::appendPoseVec(roadCentres_.data.poses, newCentres);
    if (!advanced_) {
      if (found) {
        if (Processing::goalInTrack(getOdometry(), target, roadCentres_.data.poses)) {
          goalSet_ = true;
          currentGoal_ = target;
          return true;
        } else {
          goalSet_ = false;
          return false;
        };

      } else {
        goalSet_ = false;
        status_ = pfms::PlatformStatus::IDLE;
        return false;
      }
    }
    minDist = INFINITY;
    for (int i = 0; i < roadCentres_.data.poses.size(); i++) {
      bool res = checkOriginToDestination(getOdometry(), roadCentres_.data.poses.at(i), dist, temp, estimated);
      if (dist < minDist and abs(dist) > tolerance_) {
        found = true;
        minDist = dist;
        target = roadCentres_.data.poses.at(i);
      }
    }
    if (found) {
      goalSet_ = true;
      currentGoal_ = target;
    }
  }
  return true;
}

void Ackerman::visualiseCones() {
  visualization_msgs::Marker coneMarker;
  coneMarker.ns = "cones";
  coneMarker.header.frame_id = "world";
  coneMarker.type = 3;
  coneMarker.scale.x = 0.4;
  coneMarker.scale.y = 0.4;
  coneMarker.scale.z = 0.5;
  coneMarker.color.a = 1;
  coneMarker.color.r = 1;
  coneMarker.color.g = 1;
  coneMarker.pose.orientation.x = 0;
  coneMarker.pose.orientation.y = 0;
  coneMarker.pose.orientation.z = 0;
  coneMarker.pose.orientation.w = 1;

  while (platformRunning_) {
    nav_msgs::Odometry currentLocation = getOdometry();
    double yaw = tf::getYaw(currentLocation.pose.pose.orientation);
    currentLocation.pose.pose.position.x += laserOffset_ * cos(yaw);
    currentLocation.pose.pose.position.y += laserOffset_ * sin(yaw);
    laserData_.mtx.lock();
    sensor_msgs::LaserScan scan = laserData_.data;
    laserData_.mtx.unlock();

    std::vector<geometry_msgs::Pose> cones;
    std::vector<geometry_msgs::Pose> misc;

    bool res = Processing::detectCones(currentLocation, scan, cones, miscObjects_);
    cones_.mtx.lock();
    Processing::appendPoseVec(cones_.data.poses, cones);

    visualization_msgs::MarkerArray coneMarkers;

    for (int i = 0; i < cones_.data.poses.size(); i++) {
      coneMarker.header.stamp = ros::Time::now();
      coneMarker.header.seq = cmd_pipe_seq_++;
      coneMarker.id = i;
      coneMarker.pose = cones_.data.poses.at(i);
      coneMarkers.markers.push_back(coneMarker);
    }

    cones_.mtx.unlock();
    mtx_.lock();
    visPub_.publish(coneMarkers);
    mtx_.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
};

void Ackerman::visualiseRoadCentres(void) {
  visualization_msgs::MarkerArray roadMarkers;
  visualization_msgs::Marker roadMarker;
  roadMarker.ns = "road";
  roadMarker.header.frame_id = "world";
  roadMarker.type = 1;
  roadMarker.scale.x = 0.5;
  roadMarker.scale.y = 0.5;
  roadMarker.scale.z = 0.5;
  roadMarker.color.a = 0.7;
  roadMarker.color.r = 0;
  roadMarker.color.g = 0;
  roadMarker.color.b = 1;

  while (platformRunning_) {
    {
      roadMarkers.markers.clear();
      std::lock_guard<std::mutex> lk1(cones_.mtx);
      std::lock_guard<std::mutex> lk2(roadCentres_.mtx);
      std::vector<geometry_msgs::Pose> centres;

      for (int i = 0; i < roadCentres_.data.poses.size(); i++) {
        roadMarker.header.stamp = ros::Time::now();
        roadMarker.header.seq = cmd_pipe_seq_++;
        roadMarker.id = i;
        roadMarker.pose = roadCentres_.data.poses.at(i);
        roadMarkers.markers.push_back(roadMarker);
      }
    }
    mtx_.lock();
    visPub_.publish(roadMarkers);
    mtx_.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}

sensor_msgs::LaserScan Ackerman::getScan() {
  std::lock_guard<std::mutex> lk(laserData_.mtx);
  return laserData_.data;
}
