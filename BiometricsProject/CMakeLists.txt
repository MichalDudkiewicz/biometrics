# CMakeList.txt: projekt CMake dla elementu BiometricsProject, dołącz źródło i zdefiniuj
# tutaj logikę specyficzną dla projektu.
#
cmake_minimum_required (VERSION 3.8)

# Dodaj źródło do pliku wykonywalnego tego projektu.
add_library(
	EasyBMP
	EasyBMP/EasyBMP.h
	EasyBMP/EasyBMP.cpp
)

add_executable (BiometricsProject "BiometricsProject.cpp" "BiometricsProject.h")

target_link_libraries(BiometricsProject PRIVATE EasyBMP)

# TODO: Dodaj testy i zainstaluj elementy docelowe w razie potrzeby.
