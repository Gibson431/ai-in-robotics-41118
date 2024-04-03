#include "controller.h"
#include "ros/node_handle.h"
#include "std_msgs/Float64.h"
#include <cmath>
#include <mutex>
#include <pfms_types.h>

/**
 * \brief Shared functionality/base class for platform controllers
 *
 */
Controller::Controller() {
  goalSet_ = false;
  cmd_pipe_seq_ = 0;
  std::lock_guard<std::mutex> lk(stats_.mtx);
  stats_.distance_travelled = 0;
  stats_.time_travelled = 0;
  stats_.distance_to_goal = 0;
  stats_.time_to_goal = 0;
};

Controller::Controller(ros::NodeHandle nh) : Controller() { nh_ = nh; }

Controller::~Controller() {
  platformRunning_ = false;
  runningThread_->join();
}

// // We would now have to sacrifice having a return value to have a setGoal
// // To allow to set a goal via topic we forfit having a return value
// // Point stamped can be supplied via command line
// // bool Controller::setGoal(geometry_msgs::Point goal) {
// void Controller::setGoal(const geometry_msgs::Point::ConstPtr &msg) {
//   goal_.location = *msg;
//   goalSet_ = true;
// }

bool Controller::setTolerance(double tolerance) {
  tolerance_ = tolerance;
  return true;
}

double Controller::distanceToGoal(void) {
  std::lock_guard<std::mutex> lk(stats_.mtx);
  return stats_.distance_to_goal;
}
double Controller::timeToGoal(void) {
  std::lock_guard<std::mutex> lk(stats_.mtx);
  return stats_.time_to_goal;
}
double Controller::distanceTravelled(void) {
  std::lock_guard<std::mutex> lk(stats_.mtx);
  return stats_.distance_travelled;
}
double Controller::timeInMotion(void) {
  std::lock_guard<std::mutex> lk(stats_.mtx);
  return stats_.time_travelled;
}

bool Controller::goalReached() {
  std::lock_guard<std::mutex> lk(odo_.mtx);
  double dx = currentGoal_.position.x - odo_.data.pose.pose.position.x;
  double dy = currentGoal_.position.y - odo_.data.pose.pose.position.y;
  double dz = currentGoal_.position.z - odo_.data.pose.pose.position.z;

  return (pow(pow(dx, 2) + pow(dy, 2) + pow(dz, 2), 0.5) < tolerance_);
}

// Do we need below ... maybe we can use it and impose a mutex
// To secure data?
nav_msgs::Odometry Controller::getOdometry(void) {
  std::unique_lock<std::mutex> lk(odo_.mtx);
  return odo_.data;
}

bool Controller::setOdometry(nav_msgs::Odometry odo) {
  std::lock_guard<std::mutex> lk(odo_.mtx);
  odo_.data = odo;
  return true;
}

pfms::PlatformStatus Controller::getStatus(void) {
  std::unique_lock<std::mutex> lk(odo_.mtx);
  return status_;
}

bool Controller::addGoals(std::vector<geometry_msgs::Pose> goals) {
  std::lock_guard<std::mutex> lk(goals_.mtx);
  goals_.data.insert(std::end(goals_.data), std::begin(goals), std::end(goals));
  return true;
}

int Controller::goalsRemaining() {
  std::lock_guard<std::mutex> lk(goals_.mtx);
  return goals_.data.size();
}

// pfms::PlatformType Controller::getPlatformType(void){
//     return type_;
// }
// void Controller::odoCallback(const nav_msgs::Odometry::ConstPtr &msg) {
//   std::unique_lock<std::mutex> lk(odo_.mtx);
//   odo_.data = *msg;
// }
//
// void Controller::goalsCallback(const geometry_msgs::PoseArrayConstPtr &pose_array_msg) {
//   std::unique_lock<std::mutex> lk(goals_.mtx);
//   goals_.data = pose_array_msg->poses;
// };
