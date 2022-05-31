#include <aquila/aquila.h>
#include <aquila/tools/TextPlot.h>

Aquila::DtwDataType ProcessSound(Aquila::SignalSource soundToProcess, size_t sampleSize)
{
    //okienkowanie
    Aquila::HammingWindow hamming(soundToProcess.getSamplesCount());
    soundToProcess *= hamming;

    //podzia³ na fragmenty
    Aquila::FramesCollection frameCollection(soundToProcess, sampleSize);

    //preemfaza ?
    float alpha = 0.95;
    for (int i = 0; i < frameCollection.count()-1; i++)
    {
        frameCollection.frame(i + 1) += (-alpha * frameCollection.frame(i));
    }

    //wykonanie MFCC na ka¿dym fragmencie
    Aquila::DtwDataType dtwdt;
    Aquila::Mfcc mfcc0(sampleSize);
    for (int i = 0; i < frameCollection.count(); i++)
    {
        Aquila::Frame frame = frameCollection.frame(i);
        std::vector<double> mfccValues = mfcc0.calculate(frame);
        dtwdt.push_back(mfccValues);
    }

    //obliczone wspó³czynniki MFCC zwracamy w formie wektora
    return dtwdt;
}

int main()
{
    // input signal parameters
    //const std::size_t SIZE = 64;
    const Aquila::FrequencyType f_lp = 500;

    Aquila::WaveFile sound_1("100hz.wav");

    const std::size_t SIZE = 65536;
    const Aquila::FrequencyType sampleFreq = sound_1.getSampleFrequency();

    auto fft = Aquila::FftFactory::getFft(SIZE);
    Aquila::SpectrumType spectrum = fft->fft(sound_1.toArray());

    // generowanie spektrum filtru
    Aquila::SpectrumType filterSpectrum(SIZE);
    for (std::size_t i = 0; i < SIZE; ++i)
    {
        if (i < (SIZE * f_lp / sampleFreq))
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

    double x1[SIZE];
    fft->ifft(spectrum, x1);
    
    Aquila::SignalSource lowPassSound(x1,SIZE,sampleFreq);
    
    Aquila::WaveFile::save(lowPassSound,"newSound.wav");

    //podzia³ na fragmenty
    //Aquila::FramesCollection frameCollection(sound_1, SIZE);
    int crossingPoints = 0;

    for (int i = 0; i < SIZE-1; i++)
    {
        if (!signbit(x1[i + 1]) != !signbit(x1[i])) 
            crossingPoints++;
    }


    float signalLength = lowPassSound.getSamplesCount() / lowPassSound.getSampleFrequency();

    int oscillations = crossingPoints / 2;

    int calculatedFrequency = oscillations / signalLength;

    //std::cout << oscillations << std::endl;

    if(calculatedFrequency > 60 && calculatedFrequency < 180)
        std::cout << "Speaker gender is Male and frequency is: " << calculatedFrequency << std::endl;
    else if(calculatedFrequency > 180 && calculatedFrequency < 300)
        std::cout << "Speaker gender is Female and frequency is: " << calculatedFrequency << std::endl;
    else
    std::cout << "Frequency is: " << calculatedFrequency << std::endl;

    return 0;
}

//https://medium.com/the-seekers-project/algorithmic-frequency-pitch-detection-series-part-1-making-a-simple-pitch-tracker-using-zero-9991327897a4 - ZCR usage
//https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3293852/ - voice frequencies
//https://aquila-dsp.org/articles/updated-frequency-domain-filtering-example/ - low pass filter implementation