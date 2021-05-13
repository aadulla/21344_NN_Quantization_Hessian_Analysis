# src files
set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/net.cpp
    )

# includes
include_directories("${CMAKE_SOURCE_DIR}")

# define executable with c++14
add_executable(${EXEC_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${SRC_FILES})
set_property(TARGET ${EXEC_NAME} PROPERTY CXX_STANDARD 14)

# define library with c++14
add_library(${LIB_NAME} ${SRC_FILES})
set_property(TARGET ${LIB_NAME} PROPERTY CXX_STANDARD 14)

# link with libtorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(${EXEC_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${LIB_NAME} "${TORCH_LIBRARIES}")

# link with eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(${EXEC_NAME} Eigen3::Eigen)
target_link_libraries(${LIB_NAME} Eigen3::Eigen)

# link with json
find_package(nlohmann_json 3.2.0 REQUIRED)
target_link_libraries(${EXEC_NAME} nlohmann_json::nlohmann_json)
target_link_libraries(${LIB_NAME} nlohmann_json::nlohmann_json)