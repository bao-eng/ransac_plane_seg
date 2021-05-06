cmake_minimum_required(VERSION 3.9)
project(cilantro_plane)

find_package(cilantro)

add_executable(cilantro_plane main.cpp)
target_link_libraries(cilantro_plane ${cilantro_LIBRARIES})
