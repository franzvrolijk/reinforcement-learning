using ReinforcementLearning;

namespace RL.Test
{
    public class NetworkTests
    {
        [Fact]
        public void NetworkTest()
        {
            int[] layerSizes = [2, 3, 2];

            decimal[][][] weights =
            [
                [
                    [1, 10, 1],
                    [1, 1, 1]
                ],
                [
                    [1, 1],
                    [1, 3],
                    [1, 2]
                ]
            ];

            var network = new Network(layerSizes, x => x * 2, weights);

            var result = network.Propagate([1, 1]);
            
            decimal[] expectedOutput = [60, 156];

            Assert.Equal(expectedOutput, result);
        }
    }
}