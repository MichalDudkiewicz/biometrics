cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)

set(Fingerprint "BiometricsProject_Fingerprint")
set(VoiceRecognition "BiometricsProject_VoiceRecognition")

project (BiometricsProject)

if(BUILD_FINGERPRINT)
	add_subdirectory (BiometricsProject_Fingerprint)
endif()

add_subdirectory (BiometricsProject_VoiceRecognition)
