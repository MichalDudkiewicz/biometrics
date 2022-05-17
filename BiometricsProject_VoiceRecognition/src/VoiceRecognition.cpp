#include <aquila/aquila.h>

int main()
{
    // input signal parameters
    const std::size_t SIZE = 64;
    const Aquila::FrequencyType sampleFreq = 2000;
    const Aquila::FrequencyType f1 = 125, f2 = 700;

    Aquila::WaveFile sound("E.wav");
    
    //Aquila::WaveFileHandler reader("gen.wav");
    //reader.save(sound);

    Aquila::SineGenerator sineGenerator1 = Aquila::SineGenerator(sampleFreq);
    sineGenerator1.setAmplitude(32).setFrequency(f1).generate(SIZE);
    Aquila::SineGenerator sineGenerator2 = Aquila::SineGenerator(sampleFreq);
    sineGenerator2.setAmplitude(8).setFrequency(f2).setPhase(0.75).generate(SIZE);
    auto sum = sineGenerator1 + sineGenerator2;
    Aquila::TextPlot plt("Input signal");
    plt.plot(sum);
    
    //reader.save(sound);
    Aquila::WaveFile::save(sound, "test.wav");

    // calculate the FFT
    auto fft = Aquila::FftFactory::getFft(SIZE);
    Aquila::SpectrumType spectrum = fft->fft(sound.toArray());

    plt.setTitle("Spectrum");
    plt.plotSpectrum(spectrum);

    return 0;
}