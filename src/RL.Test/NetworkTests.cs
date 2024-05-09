using ReinforcementLearning;

namespace RL.Test;

public class NetworkTests
{
    [Fact]
    public void NetworkTest()
    {
        int[] layerSizes = [2, 2, 2];

        double[][][] weights =
        [
            [
                [0.5, 1],
                [1, 0.5]
            ],
            [
                [0.5, 1],
                [1, 0.5]
            ]
        ];

        double[][][] biases =
        [
            [
                [0.5, 1],
                [1, 0.5]
            ],
            [
                [0.5, 1],
                [1, 0.5]
            ]
        ];

        var network = new Network(layerSizes, x => x * 2, weights, biases);

        var result = network.Propagate([1, 1]);
            
        double[] expectedOutput = [21, 21];

        Assert.Equal(expectedOutput, result);
    }
}