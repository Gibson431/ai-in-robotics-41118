#include <climits>
#include <gtest/gtest.h>
#include <vector>

#include <ros/package.h> //This tool allows to identify the path of the package on your system
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include "Processing/processing.h"
#include "tf/transform_datatypes.h" //To use getYaw function from the quaternion of orientation
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>

TEST(GoalLocation, TestGoalInTrack) {

  //! Below command allows to find the folder belonging to a package
  std::string path = ros::package::getPath("a3_skeleton");
  // Now we have the path, the images for our testing are stored in a subfolder
  // /test/samples
  path += "/test/bag/";
  std::string file = path + "clear-path.bag";

  //! Manipulating rosbag, from: http://wiki.ros.org/rosbag/Code%20API
  rosbag::Bag bag;
  bag.open(file); // BagMode is Read by default
  sensor_msgs::LaserScan::ConstPtr laserScan = nullptr;
  nav_msgs::Odometry::ConstPtr odom = nullptr;

  //! The bag has all the messages, so we go through all of them to find the
  //! mesages we need
  for (rosbag::MessageInstance const m : rosbag::View(bag)) {
    //! We will go through the bag and extract the laser scan and odometry
    //! We have to try to instatitate each message type

    if (m.getTopic() == "/orange/laser/scan") {
      if (laserScan == nullptr) {
        laserScan = m.instantiate<sensor_msgs::LaserScan>();
      }
    }
    if (m.getTopic() == "/ugv_odom") {
      if (odom == nullptr) {
        odom = m.instantiate<nav_msgs::Odometry>();
      }
    }
    if ((laserScan != nullptr) && (odom != nullptr)) {
      //! Now we have a laserScan and odometry so we can proceed
      //! We could also check here if we have High Intensity readings before
      //! abandoning the loop
      break;
    }
  }
  bag.close();

  ASSERT_NE(laserScan, nullptr); // Check that we have a laser scan from the bag
  ASSERT_NE(odom, nullptr);      // Check that we have a laser scan from the bag

  ////////////////////////////////////////////
  // Our code is tested below

  std::vector<geometry_msgs::Pose> cones;
  std::vector<geometry_msgs::Pose> miscObjects;
  std::vector<geometry_msgs::Pose> centres;
  bool res = Processing::detectCones(*odom, *laserScan, cones, miscObjects);
  res = Processing::findConePairCentres(*odom, cones, centres);

  geometry_msgs::Pose goal;
  goal.position.x = 36;
  goal.position.y = 13;
  res = Processing::goalInTrack(*odom, goal, centres);
  ASSERT_EQ(res, true);

  goal.position.x = 32;
  goal.position.y = 14;
  res = Processing::goalInTrack(*odom, goal, centres);
  ASSERT_EQ(res, true);

  goal.position.x = 30;
  goal.position.y = 10;
  res = Processing::goalInTrack(*odom, goal, centres);
  ASSERT_EQ(res, true);
}

TEST(GoalLocation, TestGoalNotInTrack) {

  //! Below command allows to find the folder belonging to a package
  std::string path = ros::package::getPath("a3_skeleton");
  // Now we have the path, the images for our testing are stored in a subfolder
  // /test/samples
  path += "/test/bag/";
  std::string file = path + "clear-path.bag";

  //! Manipulating rosbag, from: http://wiki.ros.org/rosbag/Code%20API
  rosbag::Bag bag;
  bag.open(file); // BagMode is Read by default
  sensor_msgs::LaserScan::ConstPtr laserScan = nullptr;
  nav_msgs::Odometry::ConstPtr odom = nullptr;

  //! The bag has all the messages, so we go through all of them to find the
  //! mesages we need
  for (rosbag::MessageInstance const m : rosbag::View(bag)) {
    //! We will go through the bag and extract the laser scan and odometry
    //! We have to try to instatitate each message type

    if (m.getTopic() == "/orange/laser/scan") {
      if (laserScan == nullptr) {
        laserScan = m.instantiate<sensor_msgs::LaserScan>();
      }
    }
    if (m.getTopic() == "/ugv_odom") {
      if (odom == nullptr) {
        odom = m.instantiate<nav_msgs::Odometry>();
      }
    }
    if ((laserScan != nullptr) && (odom != nullptr)) {
      //! Now we have a laserScan and odometry so we can proceed
      //! We could also check here if we have High Intensity readings before
      //! abandoning the loop
      break;
    }
  }
  bag.close();

  ASSERT_NE(laserScan, nullptr); // Check that we have a laser scan from the bag
  ASSERT_NE(odom, nullptr);      // Check that we have a laser scan from the bag

  ////////////////////////////////////////////
  // Our code is tested below

  std::vector<geometry_msgs::Pose> cones;
  std::vector<geometry_msgs::Pose> miscObjects;
  std::vector<geometry_msgs::Pose> centres;
  bool res = Processing::detectCones(*odom, *laserScan, cones, miscObjects);
  ASSERT_EQ(res, true);
  res = Processing::findConePairCentres(*odom, cones, centres);
  ASSERT_EQ(res, true);

  geometry_msgs::Pose goal;
  goal.position.x = 38;
  goal.position.y = 4;
  res = Processing::goalInTrack(*odom, goal, centres);
  ASSERT_EQ(res, false);

  goal.position.x = 34;
  goal.position.y = 20;
  res = Processing::goalInTrack(*odom, goal, centres);
  ASSERT_EQ(res, false);

  goal.position.x = 41;
  goal.position.y = 8;
  res = Processing::goalInTrack(*odom, goal, centres);
  ASSERT_EQ(res, false);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
