cmake_minimum_required(VERSION 3.8)
project(isaacsim_legged_nav2)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch params isaac
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)

# one-time policy export (needs a Python with torch): ros2 run isaacsim_legged_nav2 export_policies.py anymal_d
install(PROGRAMS scripts/export_policies.py
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(legged_sim_unit_tests test)
endif()

ament_package()
