#include "processing.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Quaternion.h"
#include "nav_msgs/Odometry.h"
#include "tf/LinearMath/Quaternion.h"
#include "tf/transform_datatypes.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <utility>
#include <vector>

using namespace std;

double Processing::poseToPoseDist(geometry_msgs::Pose p1, geometry_msgs::Pose p2) { return std::hypot(p1.position.x - p2.position.x, p1.position.y - p2.position.y); }

bool Processing::poseCheckCoincident(geometry_msgs::Pose p1, geometry_msgs::Pose p2, double tolerance) { return poseToPoseDist(p1, p2) < tolerance; }

bool Processing::appendPoseVec(std::vector<geometry_msgs::Pose> &drain, std::vector<geometry_msgs::Pose> &source) {
  for (auto newCone : source) {
    bool isFound = false;
    for (auto foundCone : drain) {
      if (poseCheckCoincident(newCone, foundCone, 1))
        isFound = true;
    }
    if (!isFound)
      drain.push_back(newCone);
  }
  return true;
}

geometry_msgs::Pose Processing::polarToCartesian(nav_msgs::Odometry &odom, double range, double angle) {
  double yaw = tf::getYaw(odom.pose.pose.orientation);
  double x = odom.pose.pose.position.x + range * cos(angle + yaw);
  double y = odom.pose.pose.position.y + range * sin(angle + yaw);
  geometry_msgs::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = 0.0; // Assuming the z-coordinate is 0

  pose.orientation = odom.pose.pose.orientation; // Copy the orientation from the input Odometry message
  return pose;
}

bool Processing::detectCones(nav_msgs::Odometry startLocation, const sensor_msgs::LaserScan &scan, std::vector<geometry_msgs::Pose> &cones, std::vector<geometry_msgs::Pose> &miscObjects) {
  double startDist = scan.range_max;
  double startAngle = scan.angle_min;
  bool onObj = false;
  geometry_msgs::Pose cone;
  cone.position.z = 0;
  cone.orientation.x = 0;
  cone.orientation.y = 0;
  cone.orientation.z = 0;
  cone.orientation.w = 0;
  for (int i = 0; i < scan.ranges.size(); i++) {
    if (scan.ranges.at(i) < scan.range_max and scan.ranges.at(i) > scan.range_min and !isinf(scan.ranges.at(i)) and !isnan(scan.ranges.at(i))) {
      if (!onObj) {
        startDist = scan.ranges.at(i);
        startAngle = scan.angle_min + scan.angle_increment * i;
      }
      onObj = true;
      if (i == 0)
        continue;
      if (abs(scan.ranges.at(i) - scan.ranges.at(i - 1)) > 0.5) {
        if (i != 0) {
          geometry_msgs::Pose start = polarToCartesian(startLocation, startDist, startAngle);
          geometry_msgs::Pose end = polarToCartesian(startLocation, scan.ranges.at(i - 1), scan.angle_min + scan.angle_increment * i);
          double dist = std::hypot(start.position.x - end.position.x, start.position.y - end.position.y);
          if (dist < 0.6) {

            cone.position.x = (start.position.x + end.position.x) / 2;
            cone.position.y = (start.position.y + end.position.y) / 2;
            bool acceptable = true;

            // Don't use if cone is within 3 metres, ignore the cone
            for (auto obj : miscObjects)
              if (poseCheckCoincident(cone, obj, 3)) {
                acceptable = false;
                if (!poseCheckCoincident(cone, obj, 0.5))
                  miscObjects.push_back(cone);
              }

            if (acceptable)
              cones.push_back(cone);
          } else {
            if (!isinf(dist)) {
              geometry_msgs::Pose miscObj;
              miscObj.position.x = (start.position.x + end.position.x) / 2;
              miscObj.position.y = (start.position.y + end.position.y) / 2;
              std::vector<geometry_msgs::Pose> miscVec;
              miscVec.push_back(miscObj);
              appendPoseVec(miscObjects, miscVec);
            }
          }
        }
        startDist = scan.ranges.at(i);
        startAngle = scan.angle_min + scan.angle_increment * i;
      }
    } else if (onObj) {
      onObj = false;
      geometry_msgs::Pose start = polarToCartesian(startLocation, startDist, startAngle);
      geometry_msgs::Pose end = polarToCartesian(startLocation, scan.ranges.at(i - 1), scan.angle_min + scan.angle_increment * i);
      double dist = std::hypot(start.position.x - end.position.x, start.position.y - end.position.y);
      if (dist < 0.5) {
        cone.position.x = (start.position.x + end.position.x) / 2;
        cone.position.y = (start.position.y + end.position.y) / 2;
        cones.push_back(cone);
      }
    } else
      onObj = false;
  }
  return !cones.empty();
}

static bool isValidMatch(geometry_msgs::Pose target, int i, const std::vector<geometry_msgs::Pose> &potentialCones, const std::vector<geometry_msgs::Pose> &oldCones,
                         const std::vector<geometry_msgs::Pose> &centres) {
  bool correct = false;

  if (Processing::poseToPoseDist(target, potentialCones.at(i)) - 8 < 2 and Processing::poseToPoseDist(target, potentialCones.at(i)) - 8 > -1) {
    correct = true;

    geometry_msgs::Pose checkMid;
    checkMid.position.x = (target.position.x + potentialCones.at(i).position.x) / 2;
    checkMid.position.y = (target.position.y + potentialCones.at(i).position.y) / 2;

    for (int j = 0; j < oldCones.size(); j++) {
      if (target.position.x == oldCones.at(j).position.x)
        continue;
      if (Processing::poseCheckCoincident(checkMid, oldCones.at(j), 3)) {
        correct = false;
        break;
      };
    }

    for (int j = 0; j < centres.size(); j++) {
      if (target.position.x == centres.at(j).position.x)
        continue;
      if (Processing::poseCheckCoincident(checkMid, centres.at(j), 3)) {
        correct = false;
        break;
      };
    }
  }

  return correct;
}

bool Processing::findConePairCentres(nav_msgs::Odometry startLocation, std::vector<geometry_msgs::Pose> cones, std::vector<geometry_msgs::Pose> &centres) {
  std::vector<geometry_msgs::Pose> oldCones;
  std::for_each(cones.begin(), cones.end(), [&](geometry_msgs::Pose p) { oldCones.push_back(p); });
  while (cones.size() != 0 and centres.size() < 2) {

    geometry_msgs::Pose target = cones.front();
    if (poseCheckCoincident(target, startLocation.pose.pose, 4)) {
      cones.erase(cones.begin());
      continue;
    }
    bool found = false;
    int matchI = -1;

    for (int i = 1; i < cones.size(); i++) {
      if (isValidMatch(target, i, cones, oldCones, centres)) {
        found = true;
        matchI = i;
        break;
      }
    }

    geometry_msgs::Pose foundCentre;
    if (found) {
      foundCentre.position.x = (target.position.x + cones.at(matchI).position.x) / 2;
      foundCentre.position.y = (target.position.y + cones.at(matchI).position.y) / 2;
      centres.push_back(foundCentre);
      cones.erase(cones.begin() + matchI);
      cones.erase(cones.begin());
    } else {
      cones.erase(cones.begin());
    }
  }
  return !centres.empty();
}

bool Processing::checkHeadingIsClear(const nav_msgs::Odometry &odom, const sensor_msgs::LaserScan &laserScan, const int collisionRange) {
  std::vector<geometry_msgs::Pose> cones;
  std::vector<geometry_msgs::Pose> miscObjects;
  bool res = detectCones(odom, laserScan, cones, miscObjects);
  double yaw = tf::getYaw(odom.pose.pose.orientation);
  geometry_msgs::Pose collider = odom.pose.pose;
  collider.position.x += collisionRange * cos(yaw);
  collider.position.y += collisionRange * sin(yaw);

  bool isClear = true;
  for (auto misc : miscObjects)
    if (poseCheckCoincident(collider, misc, 4))
      isClear = false;

  return isClear;
};

bool Processing::goalInTrack(const nav_msgs::Odometry &odom, geometry_msgs::Pose goal, const std::vector<geometry_msgs::Pose> &trackCentres) {
  if (poseCheckCoincident(odom.pose.pose, goal, 5))
    return true;

  if (!trackCentres.empty()) {

    geometry_msgs::Pose centrePose;
    centrePose.position.x = (odom.pose.pose.position.x + trackCentres.front().position.x) / 2;
    centrePose.position.y = (odom.pose.pose.position.y + trackCentres.front().position.y) / 2;

    if (poseCheckCoincident(centrePose, goal, 5))
      return true;
  }

  for (auto centre : trackCentres) {
    if (poseCheckCoincident(centre, goal, 5))
      return true;
  }

  geometry_msgs::Pose newMid;
  std::vector<geometry_msgs::Pose> midTrackCentres;
  for (auto mid = trackCentres.begin(); mid != trackCentres.end() - 1; mid++) {
    newMid.position.x = (mid->position.x + (mid + 1)->position.x) / 2;
    newMid.position.y = (mid->position.y + (mid + 1)->position.y) / 2;
    midTrackCentres.push_back(newMid);
  }

  for (auto centre : midTrackCentres) {
    if (poseCheckCoincident(centre, goal, 5))
      return true;
  }

  return false;
}
