cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)

if (UNIX)
    set(CMAKE_CXX_FLAGS "-lstdc++fs" )
endif (UNIX)

#tutaj dodajemy cpp-ki do naszej biblioteki do morfologii
add_executable (Fingerprint
        src/BiometricsProject.cpp
         )

add_subdirectory (thirdparty)

target_include_directories(Fingerprint PUBLIC include)
target_include_directories(Fingerprint PRIVATE thirdparty)

target_link_libraries(Fingerprint PRIVATE EasyBMP)

find_package( OpenCV REQUIRED )
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

if (UNIX)
target_link_libraries(Fingerprint PRIVATE stdc++fs)
endif (UNIX)