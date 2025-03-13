cmake_minimum_required(VERSION 3.10)

project(conslaws LANGUAGES CXX)

if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
  message(SEND_ERROR "In-source builds are not allowed.")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(UNIX)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # GCC compiler
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always -Wall -Wextra -Wpedantic -Wcast-qual -Wshadow -Wundef -Wunused-macros -Wextra-semi"
    )
    set(CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -O3 -march=native -ffast-math -fmerge-all-constants -fPIC -pipe -flto=auto -fuse-ld=mold"
    )
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -O0 -ggdb3")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    # Clang or AppleClang compiler
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -fcolor-diagnostics -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic")
    set(CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -O3 -march=native -ffast-math -fmerge-all-constants -fPIC -pipe -flto=auto"
    )
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
    # Intel compiler
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fast")
  else()
    message(SEND_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()
else()
  message(SEND_ERROR "Unsupported OS.")
endif()

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps=obj -fverbose-asm -Winline") # -fopt-info

file(GLOB SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.hpp)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

add_executable(${PROJECT_NAME} ${SRCFILES})

# find_package(Threads REQUIRED)
# target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_THREAD_LIBS_INIT})

# find_package(OpenMP REQUIRED)
# target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)

# find_package(MPI REQUIRED)
# target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_CXX)

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

# find_package(Boost REQUIRED)
# target_link_libraries(${PROJECT_NAME} PUBLIC ${Boost_LIBRARIES})

option(ENABLE_TESTING "Enable the tests" ON)

if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()
