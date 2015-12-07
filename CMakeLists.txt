cmake_minimum_required(VERSION 3.2)
project(Suduku-OpenCV)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp 
		ImageHandler.h
		ImageHandler.cpp)

add_executable(Suduku-OpenCV ${SOURCE_FILES})


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake/Modules/")
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
	include_directories(${OpenCV_INCLUDE_DIRS})
	target_link_libraries(Suduku-OpenCV ${OpenCV_LIBS})
else() 
	MESSAGE(FATAL_ERROR "OpenCV not found")
endif()

file(GLOB RESOURCE_FILES "images")
file(COPY ${RESOURCE_FILES} DESTINATION ${CMAKE_BINARY_DIR})