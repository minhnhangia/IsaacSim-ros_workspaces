cmake_minimum_required(VERSION 3.8)
project(isaacsim_spot_nav2)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch params isaac
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(spot_sim_unit_tests test)
endif()

ament_package()
