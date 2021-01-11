cmake_minimum_required(VERSION 3.9.6)
project(IARC-Vision
  LANGUAGES CXX
  VERSION   0.0.0
  DESCRIPTION "Computer Vision Module"
)

#############################
# Options
#############################
option(IARC_VISION_STATIC_BUILD "Build static library" OFF)
option(IARC_VISION_BUILD_TOOLS "Build tools" OFF)
option(IARC_VISION_BUILD_EXTERN "Build extern" OFF)

# Static or Dynamic Link
if(IARC_VISION_STATIC_BUILD)
  set(LINK_TYPE STATIC)
else()
  set(LINK_TYPE SHARED)
endif()

#############################
# Set General Variables 
#############################
# Library Variables
set(LIBRARY_NAME IARC-Vision)
set(SH_VER 0)

# Target Dir
set(IARC_VISION_LIBRARY_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Install Dir
set(IARC_VISION_LIBRARY_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/install/lib)
set(IARC_VISION_HEADER_INSTALL_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/install/inc)

#############################
# Set Compiler Flags Variable
#############################
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Base Compiler Flags
set(MY_DEBUG_FLAGS 
  -Wall 
  -Wextra
  -Werror
  -W
  -DDEBUG
  -g
)

set(MY_RELEASE_FLAGS
  -pedantic
  -03
)

#############################
# Set Files and Directories
#############################
# Source Files
set(SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/RobotDetector.cpp
)

# Header Files
set(HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/inc/RobotDetector.h
)

# Libraries
set(LIBRARIES 
  Image-Processing
  yaml-cpp
)

# Include Directory
set(PROJECT_INC_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/inc
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/inc
  ${CMAKE_CURRENT_SOURCE_DIR}/tools/inc
)

# Library Directories
set(PROJECT_LIB_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/lib
  ${CMAKE_CURRENT_SOURCE_DIR}/tools/lib
)

#############################
# Recurse into Directories
#############################
if(IARC_VISION_BUILD_EXTERN)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern)
endif()

if(IARC_VISION_BUILD_TOOLS)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tools)
endif()

# Set all Directories which contain libraries
link_directories(${PROJECT_LIB_DIRS})

#############################
# Make Target and Set Props
#############################
# Make Library Binary
add_library( ${LIBRARY_NAME} ${LINK_TYPE} ${SRC_FILES} )

# Set Library Properties
set_target_properties( ${LIBRARY_NAME} PROPERTIES
  VERSION ${PROJECT_VERSION}
  PUBLIC_HEADER "${HEADER_FILES}"
  ARCHIVE_OUTPUT_DIRECTORY "${IARC_VISION_LIBRARY_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${IARC_VISION_LIBRARY_OUTPUT_DIR}"
)

# Target Compiler Flags
target_compile_options( ${LIBRARY_NAME}
  PRIVATE
  "$<$<CONFIG:Debug>:${MY_DEBUG_FLAGS}>"
  "$<$<CONFIG:Release>:${MY_RELEASE_FLAGS}>"
)

#############################
# Find and Link Libraries
#############################
find_package(OpenCV REQUIRED)

# Target Include Directories
target_include_directories( ${LIBRARY_NAME}
  PUBLIC
  ${PROJECT_INC_DIRS}
)

# Link External Libraries
target_link_libraries( ${LIBRARY_NAME}
  PUBLIC
  ${OpenCV_LIBS}
  ${LIBRARIES}
  pthread
)

#############################
# Set Install Options
#############################
# Install 
install(
  TARGETS ${LIBRARY_NAME}
  ARCHIVE DESTINATION ${IARC_VISION_LIBRARY_INSTALL_DIR}
  LIBRARY DESTINATION ${IARC_VISION_LIBRARY_INSTALL_DIR}
  PUBLIC_HEADER DESTINATION ${IARC_VISION_HEADER_INSTALL_DIR}
)
