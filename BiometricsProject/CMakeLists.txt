cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-lstdc++fs" )

#tutaj dodajemy cpp-ki do naszej biblioteki do morfologii
add_executable (${PROJECT_NAME}
        src/BiometricsProject.cpp
         )

add_subdirectory (thirdparty)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PRIVATE thirdparty)

target_link_libraries(${PROJECT_NAME} PRIVATE EasyBMP)

find_package( OpenCV REQUIRED )
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)