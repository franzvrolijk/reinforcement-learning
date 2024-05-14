using FluentAssertions;
using ILGPU;
using ReinforcementLearning;

namespace RL.Test;

public class NetworkTests
{
    [Fact]
    public void NetworkTest()
    {
        int[] layerSizes = [2, 2, 2];
        double[] weights = [0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5];
        double[] biases = [0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5];

        var network = new Network(layerSizes, x => x * 2, weights, biases);

        var result = network.Propagate([1, 1]);
            
        double[] expectedOutput = [21, 21];

        result.Should().Equal(expectedOutput);
    }

    [Fact]
    public void LearningTest()
    {
        int[] layerSizes = [2, 2];
        double[] weights = [1, 1, 1, 1];
        double[] biases = [0, 0, 0, 0];

        var network = new Network(layerSizes, x => x, weights, biases);

        const double learningRate = 0.1d;
        const double delta = 0.00000001d;

        // Loss will decrease by 0.1 after every call
        var loss = 1d;
        double LossFunction() => loss -= 0.1;

        // First weight should see loss = 1 before adjustment, then loss = 0.9 after adjustment.
        // Change in loss will be -0.1, so gradient for first weight will be -0.1 / delta = -0.1 / 0.00000001 = -10 million
        // Gradient is negative, so weight should be adjusted by learningRate in positive direction.
        // Weight initialized to 1 -> adjusted by learningRate (+0.1) -> new weight should be 1.1

        var weightBeforeTraining = network.Weights[0];

        network.GradientDescent(LossFunction, learningRate, delta);

        var weightAfterTraining = network.Weights[0];

        var floatPrecision = 0.000000000000001;

        weightBeforeTraining.Should().BeApproximately(1, floatPrecision);
        weightAfterTraining.Should().BeApproximately(1.1, floatPrecision);
    }

    [Fact]
    public void GpuLearningTest()
    {
        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(false).CreateAccelerator(context);

        int[] layerSizes = [2, 2];
        double[] weights = [1, 1, 1, 1];
        double[] biases = [0, 0, 0, 0];

        var network = new Network(layerSizes, x => x, weights, biases);

        const double learningRate = 0.1d;
        const double delta = 0.00000001d;

        // Loss will decrease by 0.1 after every call
        var loss = 1d;
        double LossFunction() => loss -= 0.1;

        // First weight should see loss = 1 before adjustment, then loss = 0.9 after adjustment.
        // Change in loss will be -0.1, so gradient for first weight will be -0.1 / delta = -0.1 / 0.00000001 = -10 million
        // Gradient is negative, so weight should be adjusted by learningRate in positive direction.
        // Weight initialized to 1 -> adjusted by learningRate (+0.1) -> new weight should be 1.1

        var weightBeforeTraining = network.Weights[0];

        network.GradientDescentGpu(LossFunction, learningRate, delta, accelerator);

        var weightAfterTraining = network.Weights[0];

        var floatPrecision = 0.000000000000001;

        weightBeforeTraining.Should().BeApproximately(1, floatPrecision);
        weightAfterTraining.Should().BeApproximately(1.1, floatPrecision);
    }
}