cmake_minimum_required(VERSION 3.12.0)
project(main VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(main main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# find_package(Python)
add_definitions(-DHAVE_CBLAS=1)
add_definitions(-lcblas)

find_package(xtensor-blas)
target_link_libraries(main PRIVATE xtensor-blas)

find_package(xtensor-python REQUIRED)
target_link_libraries(main PRIVATE xtensor-python)

find_package(pybind11 REQUIRED CONFIG)  # or add_subdirectory(pybind11)
# add_subdirectory(pybind11)
target_link_libraries(main PRIVATE pybind11::embed)

pybind11_add_module(CppModule MODULE main.cpp)

find_package(xsimd REQUIRED)
add_definitions(-DXTENSOR_ENABLE_XSIMD)
set(XTENSOR_USE_XSIMD 1)
find_package(BLAS)
target_link_libraries(main PRIVATE ${BLAS_LIBRARIES})
# set(CMAKE_VERBOSE_MAKEFILE on)
# target_link_libraries(main PRIVATE xsimd)

# target_compile_options(main PRIVATE /arch:AVX2)
# target_compile_options(main PRIVATE -march=native)
find_package(xtensor REQUIRED)
target_link_libraries(main PRIVATE xtensor::optimize)

SET(GCC_COVERAGE_COMPILE_FLAGS "-mavx2 -O3 -ffast-math -DXTENSOR_USE_XSIMD -lcblas -DNDEBUG -march=native -fopenmp -lblas -llapack -DHAVE_CBLAS=1")
add_compile_options(-mavx2 -O3 -ffast-math -DXTENSOR_USE_XSIMD -lcblas -DNDEBUG -march=native -fopenmp -lblas -llapack -DHAVE_CBLAS=1)
# SET(GCC_COVERAGE_COMPILE_FLAGS "lcblas -llapack -DHAVE_CBLAS=1")
# add_compile_options(-lcblas -llapack -DHAVE_CBLAS=1)