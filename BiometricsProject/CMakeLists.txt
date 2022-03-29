cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)

#tutaj dodajemy cpp-ki do naszej biblioteki do morfologii
add_executable (BiometricsProject
        src/BiometricsProject.cpp
         )

add_subdirectory (thirdparty)

target_include_directories(BiometricsProject PUBLIC include)
target_include_directories(BiometricsProject PRIVATE thirdparty)

target_link_libraries(BiometricsProject PRIVATE EasyBMP)

find_package( OpenCV REQUIRED )
target_include_directories(BiometricsProject PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(BiometricsProject PUBLIC ${OpenCV_LIBS})