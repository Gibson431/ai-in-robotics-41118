#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "controllerinterface.h"
#include <atomic>
#include <cmath>
#include <mutex>
#include <pfms_types.h>
#include <thread>
#include <vector>

// Instead of Pipes now we need to use Ros communication machanism and messages
// #include <pipes.h>
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseArray.h"
#include "nav_msgs/Odometry.h"
#include "ros/node_handle.h"
#include "ros/publisher.h"
#include "ros/ros.h"
#include "tf/transform_datatypes.h" //To use getYaw function from the quaternion of orientation

//! Information about the goal for the platform
// struct GoalStats {
//   //! location of goal
//   // pfms::geometry_msgs::Point location;
//   geometry_msgs::Point location;
//
//   //! distance to goal
//   double distance;
//   //! time to goal
//   double time;
// };

/**
 * \brief Shared functionality/base class for platform controllers
 *
 * Platforms need to implement:
 * - Controller::calcNewGoal (and updating GoalStats)
 * - ControllerInterface::reachGoal (and updating PlatformStats)
 * - ControllerInterface::checkOriginToDestination
 * - ControllerInterface::getPlatformType
 * - ControllerInterface::getOdometry (and updating PlatformStats.odo)
 */
class Controller : public ControllerInterface {
public:
  /**
   * Default Controller constructor, sets odometry and metrics to initial 0 as
   * well as initialising a node handle
   */
  Controller();

  /**
   * @brief Custom Controller constructor that utilises an existing node handle
   * @param[in] nh A node handle that has already been created
   */
  Controller(ros::NodeHandle nh);

  /**
   * @brief Deconstructor that terminates the thread responsible for controlling the vehicle
   */
  ~Controller();

  /**
  @brief Checks whether the platform can travel between origin and destination
  @param[in] origin The origin pose, specified as odometry for the platform
  @param[in] goal The destination pose for the platform
  @param[in,out] distance The distance [m] the platform will need to travel
  between origin and destination. If destination unreachable distance = -1
  @param[in,out] time The time [s] the platform will need to travel between
  origin and destination, If destination unreachable time = -1
  @param[in,out] estimatedGoalPose The estimated goal pose when reaching goal
  @return bool indicating the platform can reach the destination from origin
  supplied
  */
  virtual bool checkOriginToDestination(nav_msgs::Odometry origin, geometry_msgs::Pose goal, double &distance, double &time, nav_msgs::Odometry &estimatedGoalPose) = 0;

  /**

      @brief Sets the tolerance value for calculations.
      This function sets the tolerance value used in various calculations. The tolerance
      value determines the precision or accuracy required for certain operations.
      @param tolerance The tolerance value to be set.
      @return True if the tolerance value was successfully set, false otherwise.
      */
  bool setTolerance(double tolerance);

  /**

      @brief Calculates the distance travelled.
      This function calculates the total distance travelled. The distance is measured
      based on the movement of an object or entity. The result is returned as a double value.
      @return The distance travelled as a double value.
      */
  double distanceTravelled(void);

  /**

      @brief Calculates the time spent in motion.
      This function calculates the total time spent in motion. The time is measured
      based on the duration of movement of an object or entity. The result is returned as a double value.
      @return The time spent in motion as a double value.
      */
  double timeInMotion(void);

  /**

      @brief Calculates the distance to the goal.
      This function calculates the distance from the current position to the goal position.
      The distance is measured based on the movement of an object or entity.
      The result is returned as a double value.
      @return The distance to the goal as a double value.
      */
  double distanceToGoal(void);

  /**

      @brief Calculates the time to reach the goal.
      This function calculates the estimated time to reach the goal from the current position.
      The time is measured based on the duration of movement of an object or entity.
      The result is returned as a double value.
      @return The time to reach the goal as a double value.
      */
  double timeToGoal(void);

  /**
   * @brief Updates the internal odometry
   *
   * Sometimes the pipes can give all zeros on opening, this has a little extra
   * logic to ensure only valid data is accepted
   */
  // pfms::nav_msgs::Odometry getOdometry(void);
  nav_msgs::Odometry getOdometry(void);

  /**
      @brief Sets the odometry information.
      This function sets the odometry information for further calculations and tracking.
      The odometry information is provided as a nav_msgs::Odometry object.
      @param odo The odometry information to be set.
      @return True if the odometry information was successfully set, false otherwise.
      */
  bool setOdometry(nav_msgs::Odometry odo);

  /**
      @brief Adds goals to the list.
      This function adds a list of goals to the existing goals list.
      The goals are provided as a std::vector<geometry_msgs::Pose> object.
      @param goals The goals to be added.
      @return True if the goals were successfully added, false otherwise.
      */
  bool addGoals(std::vector<geometry_msgs::Pose> goals);

  /**
      @brief Gets the platform status.
      This function retrieves the current status of the platform.
      The status is returned as a pfms::PlatformStatus object.
      @return The current platform status.
      */
  pfms::PlatformStatus getStatus();

protected:
  /**
   * @brief Checks if the goal has been reached.
   *
   * Update own odometry before calling!
   * @return true if the goal is reached
   */
  bool goalReached();

  /**
   * @brief Thread safe getter for the number of goals remaining
   * @return number of goals remaining
   */
  int goalsRemaining();

  geometry_msgs::Pose currentGoal_; //!< The current goal moving to
  std::atomic<bool> goalSet_;       //!< an idicator displaying if a goal has been set
  std::mutex mtx_;                  //!< a general purpose mutex for all data within the class

  /**
   * @brief A set of data about the current path that is mutex safe
   */
  struct StatsData {
    double distance_travelled; //!< Total distance travelled for this program run
    double distance_to_goal;   //!< The distance from the current location to the goal
    double time_travelled;     //!< Total time spent travelling for this program run
    double time_to_goal;       //!< The time estimate from the current location to the goal
    std::mutex mtx;            //!< A mutex specific to the goal stats
  } stats_;

  double tolerance_ = 0.5;            //!< Radius of tolerance
  long unsigned int cmd_pipe_seq_;    //!< The sequence number of the command
  pfms::PlatformType platformType_;   //!< The platform type
  pfms::PlatformStatus status_;       //!< The current status of the platform
  std::thread *runningThread_;        //!< a handle to the main running thread
  std::atomic<bool> platformRunning_; //!< a bool for controlling the platform running state. Used to stop threads
  bool advanced = false;              //!< An indicator or the mode the platform is running

  /**
   * @brief A memory safed storage for the goals
   */
  struct GoalData {
    std::vector<geometry_msgs::Pose> data; //!< A vector of goals in the data for of poses
    std::mutex mtx;                        //!< A mutex specific to the goals vector
  } goals_;

  /**
   * @brief A memory safed storage for the odometry
   */
  struct Odom {
    nav_msgs::Odometry data; //!< The actual odometry, updated by the odometry ros topic
    std::mutex mtx;          //!< A mutext specific to the odometry
  } odo_;

  ros::NodeHandle nh_;      //!< A handle to the running ros node
  ros::Subscriber odoSub_;  //!< A handle to the odometry Subscriber
  ros::Subscriber goalSub_; //!< A handle to the goal Subscriber

  /**
   * @brief The virtual function to be implemented as a callback to the odometry topic
   */
  virtual void odoCallback(const nav_msgs::Odometry::ConstPtr &msg) = 0;

  /**
   * @brief The virtual function to be implemented as a callback to the goals topic
   */
  virtual void goalsCallback(const geometry_msgs::PoseArrayConstPtr &pose_array_msg) = 0;

  /**
   * @brief A virtual function that initialises all the platform specific topics
   * @return whether the topics were all created properly or not
   */
  virtual bool initTopics() = 0;
};

#endif // CONTROLLER_H
