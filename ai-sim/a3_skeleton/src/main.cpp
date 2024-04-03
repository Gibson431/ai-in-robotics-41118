
#include "ros/ros.h"
// #include "sample.h"
#include "Controller/ackerman.h"
#include <thread>

/**
 * @brief An example implementation for the ackerman vehicle
 */
int main(int argc, char **argv) {

  ros::init(argc, argv, "a3_skeleton");

  ros::NodeHandle nh;

  ros::NodeHandle pn("~");
  bool advanced = false;
  pn.param<bool>("advanced", advanced, false);

  /**
   * Let's start seperate thread first, to do that we need to create object
   * and thereafter start the thread on the function desired
   */
  // std::shared_ptr<Sample> sample(new Sample(nh));
  // std::thread t(&Sample::seperateThread,sample);
  std::cout << __FUNCTION__ << " " << advanced << std::endl;
  std::shared_ptr<Ackerman> ack(new Ackerman(nh, advanced));
  // std::thread t(&Ackerman::seperateThread, ack);
  ros::spin();

  ros::shutdown();

  // t.join();

  return 0;
}
