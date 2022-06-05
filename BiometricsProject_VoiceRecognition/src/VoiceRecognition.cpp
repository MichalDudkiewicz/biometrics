#include <filesystem>
#include <aquila/aquila.h>
#include <aquila/tools/TextPlot.h>

constexpr std::size_t FrameSize = 1024;

int ZCR(Aquila::Frame signal)
{
    int crossingPoints = 0;
    for (int i = 0; i < signal.getSamplesCount() - 1; i++)
    {
        if (!signbit(signal.toArray()[i + 1]) != !signbit(signal.toArray()[i]))
        crossingPoints++;
    }
    return crossingPoints;
}

float CalculateFrequencyFromCrossingPoints(int crossingPoints, std::size_t samplesCount, Aquila::FrequencyType sampleFrequency)
{
    float signalLength = samplesCount / sampleFrequency;

    int oscillations = crossingPoints / 2;

    return oscillations / signalLength;
}

float Median(std::vector<float> values)
{
    size_t size = values.size();

    if (size == 0)
    {
        return 0;  // Undefined
    }
    else
    {
        sort(values.begin(), values.end());
        if (size % 2 == 0)
        {
            return (values[size / 2 - 1] + values[size / 2]) / 2;
        }
        else
        {
            return values[size / 2];
        }
    }
}

Aquila::Frame LowPassFilter(Aquila::Frame signal, Aquila::FrequencyType filter, Aquila::FrequencyType sampleFrequency, Aquila::SignalSource* CopySignal)
{
    auto fft = Aquila::FftFactory::getFft(signal.getSamplesCount());
    Aquila::SpectrumType spectrum = fft->fft(signal.toArray());

    // generowanie spektrum filtru
    Aquila::SpectrumType filterSpectrum(signal.getSamplesCount());
    for (std::size_t i = 0; i < signal.getSamplesCount(); ++i)
    {
        if (i < (signal.getSamplesCount() * filter / signal.getSampleFrequency()))
        {
            // passband
            filterSpectrum[i] = 1.0;
        }
        else
        {
            // stopband
            filterSpectrum[i] = 0.0;
        }
    }

    std::transform(
        std::begin(spectrum),
        std::end(spectrum),
        std::begin(filterSpectrum),
        std::begin(spectrum),
        [](Aquila::ComplexType x, Aquila::ComplexType y) { return x * y; }
    );

    double x1[FrameSize];
    fft->ifft(spectrum, x1);

    CopySignal = new Aquila::SignalSource(x1, signal.getSamplesCount(), signal.getSampleFrequency());
    Aquila::Frame f(*CopySignal, CopySignal->begin().getPosition(), CopySignal->end().getPosition());
    return f;
}

int main()
{
    // input signal parameters
    //const std::size_t SIZE = 64;

    int trueMaleResults = 0;
    int falseMaleResults = 0;
    int trueFemaleResults = 0;
    int falseFemaleResults = 0;
    int notClassified = 0;

    int totalFemaleSamples = 0;
    int totalMaleSamples = 0;

    const Aquila::FrequencyType f_lp = 377;

    std::vector<Aquila::WaveFile> databaseFiles;

    std::string path = "./database";

    for (const auto& file : std::filesystem::directory_iterator::directory_iterator(path))
    {
        //std::cout << file.path() << std::endl;
        databaseFiles.push_back(Aquila::WaveFile(file.path().string()));
    }

    for (Aquila::WaveFile file : databaseFiles)
    {
        //podzia³ na fragmenty - po 64 milisekundy
        Aquila::FramesCollection frameCollection(file, FrameSize);
        std::vector<float> frequencies;
        for (int i = 0; i < frameCollection.count(); i++)
        {
            Aquila::SignalSource* signal = new Aquila::SignalSource(file.getSampleFrequency());
            Aquila::Frame frame = LowPassFilter(frameCollection.frame(i), f_lp, file.getSampleFrequency(), signal);
            int crossingPoints = ZCR(frame);
            float calculatedFrequency = CalculateFrequencyFromCrossingPoints(crossingPoints, frame.getSamplesCount(), frame.getSampleFrequency());
            frequencies.push_back(calculatedFrequency);
        }


        //std::cout << Median(frequencies) << std::endl;

        float finalValue = Median(frequencies);

        if (finalValue > 60 && finalValue < 190)
        {
            if (file.getFilename().find("MAL") != std::string::npos)
                trueMaleResults++;
            else
                falseMaleResults++;
            
        }
        else if (finalValue > 190 && finalValue < 300)
        {
            if (file.getFilename().find("FEM") != std::string::npos)
                trueFemaleResults++;
            else
                falseFemaleResults++;
        }
        else
        {
            notClassified++;
        }
        if (file.getFilename().find("MAL") != std::string::npos)
            totalMaleSamples++;
        else if (file.getFilename().find("FEM") != std::string::npos)
            totalFemaleSamples++;
    }

    int totalResults = databaseFiles.size();

    float totalCorrect = ((float)trueFemaleResults + (float)trueMaleResults);

    std::cout << "Total number of checked samples: " << totalResults << std::endl;
    std::cout << "Total correct results are: " << totalCorrect << std::endl;
    std::cout << "Total accuracy is: " << totalCorrect/totalResults << std::endl << std::endl;

    std::cout << "Total number of male samples: " << totalMaleSamples << std::endl;
    std::cout << "Total correct male results are: " << trueMaleResults << std::endl;
    std::cout << "Total false male results are: " << falseMaleResults << std::endl << std::endl;

    std::cout << "Total number of female samples: " << totalFemaleSamples << std::endl;
    std::cout << "Total correct female results are: " << trueFemaleResults << std::endl;
    std::cout << "Total false female results are: " << falseFemaleResults << std::endl << std::endl;

    //if(calculatedFrequency > 60 && calculatedFrequency < 180)
    //    std::cout << "Speaker gender is Male and frequency is: " << calculatedFrequency << std::endl;
    //else if(calculatedFrequency > 180 && calculatedFrequency < 300)
    //    std::cout << "Speaker gender is Female and frequency is: " << calculatedFrequency << std::endl;
    //else
    //std::cout << "Frequency is: " << calculatedFrequency << std::endl;

    return 0;
}

//https://medium.com/the-seekers-project/algorithmic-frequency-pitch-detection-series-part-1-making-a-simple-pitch-tracker-using-zero-9991327897a4 - ZCR usage
//https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3293852/ - voice frequencies
//https://aquila-dsp.org/articles/updated-frequency-domain-filtering-example/ - low pass filter implementation