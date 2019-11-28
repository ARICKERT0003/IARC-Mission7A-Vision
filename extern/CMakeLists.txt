cmake_minimum_required(VERSION 3.9.6)
project(Image-Processing
  LANGUAGES CXX
  VERSION   0.0.0
  DESCRIPTION "Library for computer vision"
)

#############################
# Set General Variables
#############################
if(NOT EXTERN_LIBRARY_OUTPUT_DIR)
  set(EXTERN_LIBRARY_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
endif()

if(NOT EXTERN_LIBRARY_INSTALL_DIR)
  set(EXTERN_LIBRARY_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/install/lib")
endif()

if(NOT EXTERN_HEADER_INSTALL_DIR)
  set(EXTERN_HEADER_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/inc")
endif()

#############################
# Recurse into Subdirectories 
#############################
# Set spdlog Options
if(NOT SPDLOG_HEADER_DIR)
  # Set spdlog header directory
  set(SPDLOG_HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/spdlog/include/spdlog)

  # Add spdlog headers to Project
  list(APPEND PROJECT_INC_DIRS
    ${SPDLOG_HEADER_DIR}
    )

  # Spdlog header install 
  install(
    DIRECTORY ${SPDLOG_HEADERS_DIR} 
    DESTINATION ${EXTERN_HEADER_INSTALL_DIR}
  )
endif()

#YAML_CPP
if(NOT TARGET yaml-cpp)
  # Set YAML Options
  set(YAML_CPP_BUILD_TESTS OFF)
  set(YAML_CPP_INSTALL OFF)
  set(YAML_CPP_BUILD_CONTRIB OFF)
  set(YAML_BUILD_SHARED_LIBS ON)
  set(BUILD_SHARED_LIBS ON)
  set(YAML_CPP_HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/yaml-cpp/include)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${EXTERN_LIBRARY_OUTPUT_DIR})

  # Recurse into yaml-cpp
  add_subdirectory(yaml-cpp)

  # Add Project Include Directories
  list(APPEND PROJECT_INC_DIRS
    ${YAML_CPP_HEADER_DIR}
    )

  # YAML header install
  install(
    DIRECTORY ${YAML_CPP_HEADERS_DIR}/yaml-cpp
    DESTINATION ${EXTERN_HEADER_INSTALL_DIR}
  )
endif()

#IMAGE_PROCESSING
if(NOT TARGET Image-Processing)
  # Set Image-Processing Options
  set(IMGPROC_BUILD_EXTERN ON)
  set(IMGPROC_HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Image-Processing/inc)
  set(IMGPROC_LIBRARY_OUTPUT_DIR ${EXTERN_LIBRARY_OUTPUT_DIR})
  set(IMGPROC_HEADER_INSTALL_DIR ${EXTERN_HEADER_INSTALL_DIR})

  # Recurse into Image-Processing
  add_subdirectory(Image-Processing)

  # Add Project Include Directories
  list(APPEND PROJECT_INC_DIRS
    ${IMGPROC_HEADER_DIR}
    )
endif()
