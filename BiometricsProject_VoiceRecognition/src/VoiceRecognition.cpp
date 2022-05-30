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

    Aquila::WaveFile sound_1("FAML_Sa.wav");

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

    //std::vector<Aquila::SampleType> x1;
    double x1[SIZE];
    fft->ifft(spectrum, x1);
    
    Aquila::SignalSource lowPassSound(x1,SIZE,sampleFreq);
    
    Aquila::WaveFile::save(lowPassSound,"newSound.wav");

    //podzia³ na fragmenty
    Aquila::FramesCollection frameCollection(sound_1, SIZE);

    for (int i = 0; i < frameCollection.count(); i++)
    {
        std::cout << sound_1.getAudioLength() << std::endl;
    }

    //Aquila::DtwDataType data_1 = ProcessSound(sound_1, SIZE);


    //Aquila::WaveFile sound_2("MCBR_Sa.wav");

    //Aquila::DtwDataType data_2 = ProcessSound(sound_2, SIZE);

    ////porównanie dŸwiêków
    //Aquila::Dtw dtw(Aquila::euclideanDistance, Aquila::Dtw::PassType::Diagonals);
    //
    //double distance_1 = dtw.getDistance(data_1, data_2);
    //std::cout << "Distance : " << distance_1 << std::endl;


    return 0;
}