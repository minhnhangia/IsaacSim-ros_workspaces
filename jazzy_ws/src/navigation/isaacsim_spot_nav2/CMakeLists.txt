cmake_minimum_required(VERSION 3.8)
project(isaacsim_spot_nav2)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch params isaac
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS scripts/cloud_relay.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
