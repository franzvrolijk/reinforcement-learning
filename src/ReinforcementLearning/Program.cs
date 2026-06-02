using ILGPU;

namespace ReinforcementLearning;

public class Program
{
    private static void Main()
    {
        var network = new Network([4, 8, 4, 1], Activation.LeakyReLU, Activation.Sigmoid);

        TrainNetwork(network);
    }

    private static void TrainNetwork(Network network)
    {
        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(false).CreateAccelerator(context);

        const int trainingIterations = 1000;
        const double initialLearningRate = 0.001d;
        const double learningRateDecay = 0.0005d;
        const double delta = 0.00000001d;
        const double targetAccuracy = 0.98;

        var iteration = 1;
        var trainingComplete = false;
        
        while (!trainingComplete)
        {
            var learningRate = initialLearningRate / (1 + learningRateDecay * iteration);

            for (var i = 0; i < trainingIterations; i++)
            {
                network.GradientDescentGpu(() =>
                {
                    
                    return loss;
                }, learningRate, delta, accelerator);
            }
            iteration++;
        }
    }
}