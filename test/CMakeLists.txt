cmake_minimum_required(VERSION 3.10)

project(conslaws_test LANGUAGES CXX)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.14.0
                     GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(googletest)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_executable(${PROJECT_NAME} ${SRCFILES})

target_link_libraries(${PROJECT_NAME} PUBLIC gtest_main)

find_package(Eigen3 REQUIRED NO_MODULE)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

if(0)
  find_package(BLAS REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC ${BLAS_LIBRARIES})
  add_compile_definitions(EIGEN_USE_BLAS)

  find_package(LAPACK REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC ${LAPACK_LIBRARIES})

  find_package(LAPACKE REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC ${LAPACKE_LIBRARIES})
  add_compile_definitions(EIGEN_USE_LAPACKE)
endif()
