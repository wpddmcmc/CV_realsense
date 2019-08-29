cmake_minimum_required (VERSION 2.8)

project (videotest)
# using C++11 
set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} -std=c++11 -pthread")

set(OpenCV_DIR "/home/mingcongchen/app/opencv-3.4.6/build")

find_package( OpenCV REQUIRED )
find_package(OpenGL REQUIRED)
set(DEPENDENCIES realsense ${OPENGL_LIBRARIES})

find_path(GLFW_INCLUDE_DIR NAMES GLFW/glfw3.h
        PATHS /usr/X11R6/include
              /usr/include/X11
              /opt/graphics/OpenGL/include
              /opt/graphics/OpenGL/contrib/libglfw
              /usr/local/include
              /usr/include/GL
              /usr/include
    )

find_library(GLFW_LIBRARIES NAMES glfw glfw3
            PATHS /usr/lib64
                  /usr/lib
                  /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
                  /usr/local/lib64
                  /usr/local/lib
                  /usr/local/lib/${CMAKE_LIBRARY_ARCHITECTURE}
                  /usr/X11R6/lib
    )

aux_source_directory(./src/ DIR_SRCS)

add_executable(videotest ${DIR_SRCS})

include_directories (
    /usr/local/include
    /usr/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
${OpenCV_INCLUDE_DIRS})

target_link_libraries(videotest
${OpenCV_LIBS}
/usr/lib
/usr/local/lib
${DEPENDENCIES}
    )
