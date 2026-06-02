using FluentAssertions;
using ILGPU;
using ReinforcementLearning;

namespace RL.Test;

public class NetworkTests
{
    [Fact]
    public void PropagateTest()
    {
        int[] layerSizes = [2, 2, 2];
        double[] weights = [0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5];
        double[] biases = [0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5];

        var network = new Network(layerSizes, Activation.Sigmoid, Activation.Sigmoid, weights, biases);

        var result = network.Propagate([1, 1]);

        double[] expectedOutput = [0.9492548465650223, 0.9492548465650223];

        for (var i = 0; i < result.Length; i++)
        {
            result[i].Should().BeApproximately(expectedOutput[i], 0.000000000000001);
        }

        result = network.Propagate([1, 1]);

        for (var i = 0; i < result.Length; i++)
        {
            result[i].Should().BeApproximately(expectedOutput[i], 0.000000000000001);
        }
    }

    [Fact]
    public void PropagateGpuTest()
    {
        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(false).CreateAccelerator(context);

        int[] layerSizes = [2, 2, 2];
        double[] weights = [0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5];
        double[] biases = [0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5];

        var network = new Network(layerSizes, Activation.Sigmoid, Activation.Sigmoid, weights, biases);

        var result = network.PropagateGpu([1, 1], accelerator);

        double[] expectedOutput = [0.9492548465650223, 0.9492548465650223];

        for (var i = 0; i < result.Length; i++)
        {
            result[i].Should().BeApproximately(expectedOutput[i], 0.000000000000001);
        }
    }

    [Fact]
    public void GradientDescentTest()
    {
        int[] layerSizes = [2, 2];
        double[] weights = [1, 1, 1, 1];
        double[] biases = [0, 0, 0, 0];

        var network = new Network(layerSizes, x => x, x => x, weights, biases);

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
    public void GradientDescentGpuTest()
    {
        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(false).CreateAccelerator(context);

        int[] layerSizes = [2, 2];
        double[] weights = [1, 1, 1, 1];
        double[] biases = [0, 0, 0, 0];

        var network = new Network(layerSizes, x => x, x => x, weights, biases);

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

    [Fact]
    public void GetWeightIndexTest()
    {
        var network = new Network([2, 3, 4, 1], x => x, x => x);

        var weightIndex122 = network.GetWeightIndex(1, 2, 2);
        var weightIndex200 = network.GetWeightIndex(2, 0, 0);
        var weightIndex230 = network.GetWeightIndex(2, 3, 0);

        Assert.Equal(16, weightIndex122);
        Assert.Equal(18, weightIndex200);
        Assert.Equal(21, weightIndex230);
    }

    [Fact]
    public void RemoveNodeTest()
    {   
        // Remove at        |        |        |
        //                  v        v        v
        double[] weights = [1, 2, 3, 1, 3, 2, 3, 2, 1];
        double[] biases =  [1, 2, 3, 1, 3, 2, 3, 2, 1];

        var network = new Network([2, 3, 1], x => x, x => x, weights, biases);

        network.RemoveNodeFromLayer(1);

        Assert.Equal([2, 3, 3, 2, 2, 1], network.Weights);
        Assert.Equal([2, 3, 3, 2, 2, 1], network.Biases);
    }

    [Fact]
    public void AddNodeTest()
    {
        // Add at          |        |        |
        //                 v        v        v
        double[] weights = [1, 2, 3, 1, 3, 2, 3, 2, 1  ];
        double[] biases =  [1, 2, 3, 1, 3, 2, 3, 2, 1  ];

        var network = new Network([2, 3, 1], x => x, x => x, weights, biases);

        network.AddNodeToLayer(1, 10, 10);

        Assert.Equal([10, 1, 2, 3, 10, 1, 3, 2, 10, 3, 2, 1], network.Weights);
        Assert.Equal([10, 1, 2, 3, 10, 1, 3, 2, 10, 3, 2, 1], network.Biases);
    }
}