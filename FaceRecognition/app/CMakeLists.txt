# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/include)

#opencv 4.0.1
add_library(libopencv_java4 SHARED IMPORTED)
set_target_properties(libopencv_java4 PROPERTIES IMPORTED_LOCATION
                     ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopencv_java4.so)

#ncnn
add_library(libncnn STATIC IMPORTED )
set_target_properties(libncnn PROPERTIES IMPORTED_LOCATION
                     ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libncnn.a)

#local
file(GLOB SRC src/main/cpp/src/*.cpp src/main/cpp/src/*.h src/main/cpp/src/*.c src/main/cpp/src/*.hpp)
set(SRC_CODE ${SRC})
add_library(native-lib SHARED ${SRC_CODE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI})


find_library(log-lib log )


target_link_libraries(native-lib libopencv_java4 libncnn ${log-lib} )