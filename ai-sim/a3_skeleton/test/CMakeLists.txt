#############
## Testing ##
#############
## Add gtest based cpp test target and link libraries
catkin_add_gtest(${PROJECT_NAME}_test utest.cpp)
catkin_add_gtest(${PROJECT_NAME}_obstacleTest obstacleTest.cpp)
catkin_add_gtest(${PROJECT_NAME}_pathTest pathTest.cpp)
if(TARGET ${PROJECT_NAME}_test)
   target_link_libraries(${PROJECT_NAME}_test ${catkin_LIBRARIES} ${PROJECT_NAME}_controller ${PROJECT_NAME}_processing )
   target_link_libraries(${PROJECT_NAME}_obstacleTest ${catkin_LIBRARIES} ${PROJECT_NAME}_controller ${PROJECT_NAME}_processing )
   target_link_libraries(${PROJECT_NAME}_pathTest ${catkin_LIBRARIES} ${PROJECT_NAME}_controller ${PROJECT_NAME}_processing )
   message ( STATUS " catkin_LIBRARIES ${catkin_LIBRARIES}")
endif()
