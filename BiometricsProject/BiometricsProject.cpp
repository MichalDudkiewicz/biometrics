// BiometricsProject.cpp: definiuje punkt wejścia dla aplikacji.
//

#include "BiometricsProject.h"
#include "EasyBMP/EasyBMP.h"

using namespace std;

int main()
{
	cout << "Hello biometrics" << endl;
	BMP Image;
	Image.SetSize(400, 400);
	Image.WriteToFile("test.bmp");
	return 0;
}
