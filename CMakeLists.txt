cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(21344)

# executable name
set(EXEC_NAME "21344")
# library name
set(LIB_NAME "21344_lib")

# add g++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")

# build code
add_subdirectory(src)
add_subdirectory(inc)