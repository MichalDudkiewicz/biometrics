﻿cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)

#tutaj dodajemy cpp-ki do naszej biblioteki do morfologii
add_executable (VoiceRecognition
        src/VoiceRecognition.cpp
         )

target_include_directories(VoiceRecognition PUBLIC include)

target_link_libraries(VoiceRecognition PRIVATE "C:/Program Files (x86)/Aquila/lib/Aquila.lib"
                                            "C:/Program Files (x86)/Aquila/lib/Ooura_fft.lib")

target_include_directories(VoiceRecognition PUBLIC "C:/Program Files (x86)/Aquila/include")

