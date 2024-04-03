#ifndef LASERPROCESSING_H
#define LASERPROCESSING_H

#include "Controller/controller.h"
#include "nav_msgs/Odometry.h"
#include "tf/transform_datatypes.h"
#include <geometry_msgs/Pose.h>
#include <math.h>
#include <sensor_msgs/LaserScan.h>
#include <system_error>

namespace Processing {
/**
    @brief Detects cones in a laser scan and updates the corresponding lists.
    This function analyzes a given laser scan to detect cones and other miscellaneous objects.
    It updates the lists of detected cones and miscellaneous objects with their respective poses.
    The startLocation parameter represents the starting location of the robot.
    @param startLocation The starting location of the robot (Odometry message).
    @param scan The laser scan data containing range measurements.
    @param cones A vector to store the poses of detected cones.
    @param miscObjects A vector to store the poses of other miscellaneous objects.
    @return True if cones are detected, false otherwise.
    */
bool detectCones(nav_msgs::Odometry startLocation, const sensor_msgs::LaserScan &scan, std::vector<geometry_msgs::Pose> &cones, std::vector<geometry_msgs::Pose> &miscObjects);
/**
    @brief Converts polar coordinates to Cartesian coordinates in the x-y plane.
    This function takes an Odometry message, representing the robot's pose, and converts
    polar coordinates (range and angle) to Cartesian coordinates (x and y) in the x-y plane.
    The resulting pose is returned as a geometry_msgs::Pose object.
    @param odom The Odometry message containing the robot's pose.
    @param range The range or distance from the origin to the desired point.
    @param angle The angle or direction in radians from the positive x-axis to the desired point.
    @return A geometry_msgs::Pose object representing the Cartesian coordinates in the x-y plane.
    */
geometry_msgs::Pose polarToCartesian(nav_msgs::Odometry &odom, double range, double angle);
/**
    @brief Finds the centers of pairs of cones based on their poses.
    This function takes a starting location, a vector of cone poses, and updates another vector
    with the centers of the pairs of cones found. The algorithm searches for pairs of cones
    that are valid matches based on certain criteria and computes their center as the average
    position between them.
    @param startLocation The starting location of the robot (Odometry message).
    @param cones A vector containing the poses of detected cones.
    @param centres A vector to store the poses of the centers of the cone pairs.
    @return True if valid cone pairs are found, false otherwise.
    */
bool findConePairCentres(nav_msgs::Odometry startLocation, std::vector<geometry_msgs::Pose> cones, std::vector<geometry_msgs::Pose> &centres);
/**
    @brief Calculates the Euclidean distance between two poses.
    This function calculates the Euclidean distance between two poses, p1 and p2.
    The distance is computed based on the positions of the poses in the x-y plane.
    @param p1 The first pose.
    @param p2 The second pose.
    @return The Euclidean distance between the two poses.
    */
double poseToPoseDist(geometry_msgs::Pose p1, geometry_msgs::Pose p2);
/**
    @brief Checks if two poses are coincident within a given tolerance.
    This function compares two poses, p1 and p2, and determines if they are coincident within a specified tolerance.
    The tolerance is a double value that represents the maximum allowable distance between the poses.
    @param p1 The first pose to compare.
    @param p2 The second pose to compare.
    @param tolerance The maximum allowable distance between the poses.
    @return True if the poses are coincident within the tolerance, false otherwise.
    */
bool poseCheckCoincident(geometry_msgs::Pose p1, geometry_msgs::Pose p2, double tolerance);
/**
    @brief Appends the contents of a source vector to a sink vector, avoiding duplicate poses.
    This function takes a source vector and a sink vector of geometry_msgs::Pose objects.
    It iterates over each pose in the source vector and checks if it is already present in the sink vector.
    If a pose is not found in the sink vector, it is appended to the end of the sink vector.
    The function ensures that duplicate poses are not added to the sink vector.
    @param sink The vector to which the source poses will be appended.
    @param source The vector containing the poses to be appended to the sink vector.
    @return True, indicating the successful appending of poses.
    */
bool appendPoseVec(std::vector<geometry_msgs::Pose> &sink, std::vector<geometry_msgs::Pose> &source);
/**
    @brief Checks if the heading direction is clear of obstacles within a specified range.
    This function examines the laser scan data and the robot's odometry to determine if the heading
    direction is clear of obstacles. It detects cones and miscellaneous objects in the environment
    using the detectCones() function. It then extends a collider pose in the robot's heading direction
    by a specified collision range. The function checks if any miscellaneous objects coincide with
    the extended collider pose to determine if the heading direction is clear.
    @param odom The Odometry message containing the robot's pose.
    @param laserScan The laser scan data for obstacle detection.
    @param collisionRange The range for collision detection in the heading direction.
    @return True if the heading direction is clear of obstacles, false otherwise.
    */
bool checkHeadingIsClear(const nav_msgs::Odometry &odom, const sensor_msgs::LaserScan &laserScan, const int collisionRange);
/**
    @brief Checks if the goal pose is within the track or close to any track centres.
    This function determines if the goal pose is within the track or close to any of the track centres.
    It first checks if the goal pose coincides with the current robot pose within a specified tolerance.
    If they coincide, it returns true to indicate that the goal is within the track.
    If there are track centres available, it calculates the average position between the robot pose and
    the first track centre. It then checks if this averaged pose coincides with the goal pose within
    the specified tolerance. If they coincide, it returns true.
    Next, it iterates through each track centre and checks if any of them coincide with the goal pose
    within the specified tolerance. If a match is found, it returns true.
    Additionally, it calculates intermediate track centres between consecutive track centres and performs
    the same coincidence check with the goal pose. If any of the intermediate track centres coincide with
    the goal pose within the specified tolerance, it returns true.
    If none of the above conditions are met, it returns false to indicate that the goal is not within the track.
    @param odom The Odometry message containing the robot's pose.
    @param goal The goal pose to be checked.
    @param trackCentres The vector of track centres for reference.
    @return True if the goal is within the track or close to any track centres, false otherwise.
    */
bool goalInTrack(const nav_msgs::Odometry &odom, geometry_msgs::Pose goal, const std::vector<geometry_msgs::Pose> &trackCentres);
} // namespace Processing

#endif // DETECTCABINET_H
