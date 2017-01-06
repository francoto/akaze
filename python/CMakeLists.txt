project(python_bindings)

cmake_minimum_required (VERSION 2.6)
if (COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif ()


# ============================================================================ #
# Build Type. By default is release
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug, Release" FORCE)


# ============================================================================ #
# Where to output binaries and libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")


set(libraries)

# Boost
find_package(Boost REQUIRED COMPONENTS python)
list( APPEND libraries ${Boost_PYTHON_LIBRARY} )
include_directories(${Boost_INCLUDE_DIR})

# Python
find_package(PythonLibs REQUIRED)
list( APPEND libraries ${PYTHON_LIBRARIES})
include_directories(${PYTHON_INCLUDE_DIRS})

# opencv
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
list( APPEND libraries ${OpenCV_LIBS} )

include_directories(.)

# akaze
include_directories(/usr/local/akaze/include)
list( APPEND libraries /usr/local/akaze/lib/libAKAZE.a /usr/local/akaze/lib/libAKAZE_CUDA.a)

add_library(akaze_pybindings SHARED python_bindings.cpp pyboost_cv2_converter.cpp)

target_link_libraries(akaze_pybindings ${libraries})

