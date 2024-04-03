#ifndef ACKERMAN_H
#define ACKERMAN_H

#include "controller.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseArray.h"
#include "nav_msgs/Odometry.h"
#include "ros/node_handle.h"
#include "ros/service_client.h"
#include "ros/subscriber.h"
#include "sensor_msgs/LaserScan.h"
#include "std_srvs/SetBool.h"
#include <cmath>
#include <mutex>

/*!
 * @brief     Ackerman controller class
 * @details
 * The controller for the Ackerman vehicle. It performs the calculations behind
 * the movement towards goals.
 * @author    Timothy Gibson
 * @date      2023-05-30
 */

class Ackerman : public Controller {
public:
  /**
   * @brief Default constructor for the Ackerman class.
   * Calls the parameterized constructor with default arguments.
   */
  Ackerman();

  /**
   * @brief Parameterized constructor for the Ackerman class.
   * Initializes the Ackerman controller with the given ROS node handle and advanced flag.
   * Initializes member variables, sets the platform type to ACKERMAN, and starts the run thread and visualization
   * threads.
   *
   * @param nh The ROS node handle.
   * @param advanced Flag indicating if advanced mode is enabled.
   */
  Ackerman(ros::NodeHandle nh, bool advanced);

  /**
   * @brief Destructor for the Ackerman class.
   * Stops the platform, joins the visualization threads, and cleans up resources.
   */
  ~Ackerman();

  /**
  @brief Executes the main loop for the Ackerman controller.
  This function runs the main control loop for the Ackerman controller. It continuously checks the status of the platform
  and the goal set, and performs the necessary actions to reach the goal. The control loop includes calculating the distance
  to the goal, checking if the goal has been reached, checking for obstacles in the path, calculating optimal braking force,
  and sending commands to control the platform. It also updates the statistics related to distance and time travelled.
  @details
  The main control loop of this function runs as long as the platform is running. It consists of two nested while loops:
    - The outer while loop keeps running as long as the platform is running.
    - The inner while loop runs as long as the platform is not idle and either the goal is set or the advanced mode is enabled.

  Inside the inner while loop, the following steps are performed:
    - Calculate the distance to the goal using the current odometry and the current goal pose.
    - Update the estimated goal pose, distance to goal, and time to goal in the statistics.
    - Check if the goal has been reached, and if so, break out of the loop.
    - Check if the heading is clear by examining the current odometry and the scan data.
    - If any obstacles are detected, the platform is stopped and the control loop is exited.
    - Calculate the optimal braking force based on the distance to the goal and the current velocity.
    - Send the command to control the platform with the calculated throttle, steering angle, and braking force.
    - Update the statistics with the elapsed time and the distance travelled.

  After the inner while loop exits, the function checks if the platform is still not idle.
  If so, it updates the goals and sends a command to stop the platform.
  Finally, the function waits for a short duration before returning.
 */
  void run();

  /**
   * @brief Checks whether the ackerman can reach a goal based on a given origin point
   * and the goal. Estimates the distance and time required to get to the goal.
   * Estimates the pose of the ackerman when it reaches the goal.
   * @param[in] origin The origin pose, specified as odometry for the platform
   * @param[in] goal The destination point for the platform
   * @param[in,out] distance The distance [m] the platform will need to travel
   * between origin and destination. If destination unreachable, distance = INFINITY
   * @param[in,out] time The time [s] the platform will need to travel between
   * origin and destination. If destination unreachable, time = INFINITY
   * @param[in,out] estimatedGoalPose The estimated goal pose when reaching goal
   * @return bool indicating if the ackerman can reach the destination from
   * origin supplied
   */
  bool checkOriginToDestination(nav_msgs::Odometry origin, geometry_msgs::Pose goal, double &distance, double &time, nav_msgs::Odometry &estimatedGoalPose);
  /**
   * @brief Retrieves the laser scan data.
   * @details This function returns the latest laser scan data available in the `laserData_` member variable.
   * @return The laser scan data.
   */
  sensor_msgs::LaserScan getScan();

private:
  bool advanced_;                                                        //!< Toggle for advanced mode
  const double steeringRatio_ = 17.3;                                    //!< Steering ration used to convert from angle of wheel to angle of
                                                                         //!< steering wheel
  const double lockToLockRevs_ = 3.2;                                    //!< Maximum revolutions of the steering wheel
  const double maxSteerAngle_ = M_PI * lockToLockRevs_ / steeringRatio_; //!< Maximum angle of steering
  const double trackWidth_ = 1.638;                                      //!< Ackerman width from wheel contact points
  const double wheelRadius_ = 0.36;                                      //!< Radius of the ackerman's wheels
  const double wheelBase_ = 2.65;                                        //!< Length of the ackerman from contact point of front wheels to back wheels
  const double laserOffset_ = 3.7;                                       //!< The laser offset along the local x axis of the ackerman
  const double maxBrakeTorque_ = 8000.0;                                 //!< Maximum brake available in the ackerman
  const double defaultThrottle_ = 0.2;                                   //!< Default throttle for this ackerman. Produces a top speed of 5.82 m/s
  const double minRadius_ = wheelBase_ / tan(maxSteerAngle_);            //!< Minimum turning radius of this ackerman
  double throttle_ = defaultThrottle_;                                   //!< Variable throttle
  double steerAngle_ = 0.0;                                              //!< Calculated required steering angle. Calculation done after calling checkOriginToDestination

  struct LaserData {
    sensor_msgs::LaserScan data; //!< Laser scan data. */
    std::mutex mtx;              //!< Mutex for thread safety. */
  } laserData_;

  struct ConesData {
    geometry_msgs::PoseArray data; //!< Pose array representing cones data. */
    std::mutex mtx;                //!< Mutex for thread safety. */
  } cones_;

  struct RoadCentreData {
    geometry_msgs::PoseArray data; //!< Pose array representing road center data. */
    std::mutex mtx;                //!< Mutex for thread safety. */
  } roadCentres_;

  std::vector<geometry_msgs::Pose> miscObjects_; //!< Vector of miscellaneous object poses. */

  std::atomic<bool> running_;  //!< Atomic flag indicating if the program is running. */
  ros::Publisher visPub_;      //!< ROS publisher for visualization. */
  ros::Publisher brakePub_;    //!< ROS publisher for brake commands. */
  ros::Publisher steeringPub_; //!< ROS publisher for steering commands. */
  ros::Publisher throttlePub_; //!< ROS publisher for throttle commands. */
  ros::Publisher conesPub_;    //!< ROS publisher for cones data. */

  ros::ServiceServer goNoGoService_; //!< ROS service server for go/no-go decision. */

  ros::Subscriber laserSub_; //!< ROS subscriber for laser data. */

  std::thread *visConeThread_; //!< Pointer to the visualization cone thread. */
  std::thread *visRoadThread_; //!< Pointer to the visualization road thread. */
private:
  /**
   * @brief The control sequence responsible for moving the ackerman towards the
   * goals. This procedure looks ahead towards subsequent goals to determine how
   * much the ackerman should slow down before reaching the current goal.
   */
  void reachGoals();
  /**
   * @brief Callback function for the odometry subscriber.
   * Sets the current odometry data.
   *
   * @param msg The received odometry message.
   */
  void odoCallback(const nav_msgs::Odometry::ConstPtr &msg);
  /**
   * @brief Callback function for the goals subscriber.
   * Adds the received goal poses to the list of goals.
   *
   * @param pose_array_msg The received goal poses message.
   */
  void goalsCallback(const geometry_msgs::PoseArrayConstPtr &pose_array_msg);
  /**
   * @brief Callback function for the laser scan subscriber.
   * Updates the laser scan data.
   *
   * @param msg The received laser scan message.
   */
  void laserCallback(const sensor_msgs::LaserScan::ConstPtr &msg);
  /**
   * @brief Service callback function for handling the mission request.
   * Starts or stops the platform based on the request data.
   *
   * @param req The mission request data.
   * @param res The mission response data.
   * @return Always returns true.
   */
  bool request(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res);

  /**
   * @brief Initializes the topics for the Ackerman class.
   * Subscribes to necessary topics and advertises required publishers and services.
   *
   * @return True if topic initialization was successful, false otherwise.
   */
  bool initTopics();
  /**
   * @brief Sends the control commands for throttle, steering, and brake to the platform.
   *
   * @param throttle Throttle command value.
   * @param steering Steering command value.
   * @param brake Brake command value.
   * @return Always returns false.
   */
  bool sendCommand(double throttle, double steering, double brake);
  /**
   * @brief Updates the goals for the Ackerman controller.
   * @details This function updates the goals for the Ackerman controller based on the current state and environment. It
   * checks if there are any goals available and sets the goalSet_ flag accordingly. If a goal is already set and it has
   * been reached, the function removes the goal from the list of goals. If advanced mode is enabled, it finds the
   * nearest goal among the road centres. If a goal is found, it sets the goalSet_ flag and assigns the currentGoal_
   * variable to the selected goal.
   * @return True if a goal is set, False otherwise.
   */
  bool updateGoals();

  /**
   * @brief Visualizes cones in the environment.
   * @details This function visualizes cones in the environment using marker messages. It retrieves the current position
   * and orientation of the robot, adjusts the position using the laser offset, and detects cones using the provided
   * laser scan data. The detected cone poses are stored in the `cones_` member variable, and marker messages are
   * created for each cone. The markers are then published for visualization.
   */
  void visualiseCones();
  /**
   * @brief Visualizes road centres in the environment.
   * @details This function visualizes road centres in the environment using marker messages. It retrieves the current
   * road centres from the `roadCentres_` member variable and creates marker messages for each road centre. The markers
   * are then published for visualization.
   */
  void visualiseRoadCentres();
};

#endif // ACKERMAN_H
