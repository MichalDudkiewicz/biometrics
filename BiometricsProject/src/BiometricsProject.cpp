#include "BiometricsProject.h"
#include "EasyBMP/EasyBMP.h"

int main()
{
	std::cout << "Hello biometrics" << std::endl;
	BMP Image;
	Image.SetSize(400, 400);
	Image.WriteToFile("test.bmp");
	return 0;
}
