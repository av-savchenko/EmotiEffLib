cmake_minimum_required(VERSION 3.10.2)
project("emotiefflib" VERSION 1.1.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(OpenCV REQUIRED)

option(BUILD_TESTS "Build EmotiEffCppLib with tests" OFF)
option(WITH_TORCH "Path to the direcotry with Libtorch" OFF)
option(WITH_ONNX "Path to the direcotry with ONNXRuntime" OFF)
option(BUILD_SHARED_LIBS "Build shared library instead of static" OFF)

if(NOT WITH_TORCH AND NOT WITH_ONNX)
    message(FATAL_ERROR "At least one inference engine should be specified: -DWITH_TORCH or -DWITH_ONNX")
endif()

set(ORIG_VALUE_BUILD_TESTS ${BUILD_TESTS})
set(BUILD_TESTS OFF)
set(CPP17 ON)
add_subdirectory("${PROJECT_SOURCE_DIR}/3rdparty/xtl")
add_subdirectory("${PROJECT_SOURCE_DIR}/3rdparty/xtensor")
set(BUILD_TESTS ${ORIG_VALUE_BUILD_TESTS})

file(GLOB src_files
    "src/*.cpp"
)

set(LINK_LIBRARIES ${OpenCV_LIBS})
set(INCLUDE_DIRS
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/3rdparty/xtensor/include"
)

if (WITH_TORCH)
    list(APPEND CMAKE_PREFIX_PATH "${WITH_TORCH}")
    find_package(Torch REQUIRED)
    list(APPEND LINK_LIBRARIES ${TORCH_LIBRARIES})
    file(GLOB torch_src_files
        "src/backends/torch/*.cpp"
    )
    list(APPEND src_files ${torch_src_files})
    add_definitions(-DWITH_TORCH)
endif()

if (WITH_ONNX)
    list(APPEND CMAKE_PREFIX_PATH "${WITH_ONNX}")
    find_package(onnxruntime REQUIRED)
    list(APPEND INCLUDE_DIRS "${WITH_ONNX}/include")
    find_library(ONNX_RUNTIME_LIB onnxruntime HINTS "${WITH_ONNX}/lib")
    list(APPEND LINK_LIBRARIES ${ONNX_RUNTIME_LIB})
    file(GLOB onnx_src_files
        "src/backends/onnx/*.cpp"
    )
    list(APPEND src_files ${onnx_src_files})
    add_definitions(-DWITH_ONNX)
endif()

include_directories(
    ${INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
    ${src_files}
)

# This automatically sets STATIC or SHARED based on the option
set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION 1.0
    SOVERSION 1
)

target_link_libraries(${PROJECT_NAME}
    ${LINK_LIBRARIES}
    xtensor
)

if(${BUILD_TESTS})
    include(FetchContent)
    FetchContent_Declare(
      googletest
      # Release v1.16.0
      URL https://github.com/google/googletest/archive/6910c9d9165801d8827d628cb72eb7ea9dd538c5.zip
    )
# For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
endif()
