using System.Diagnostics;

namespace ReinforcementLearning;

public class Program
{
    private static async Task Main()
    {
        var numNetworks = 300;
        var boardSize = 10;
        var numIter = 15;
        var generation = 1;

        var networks = Enumerable.Range(0, numNetworks)
            .Select(_ => new Network([4, 8, 16, 8, 4], Sigmoid))
            .ToList();

        var stopwatch = new Stopwatch();
        stopwatch.Start();

        while (true)
        {
            var networkScoreDict = new Dictionary<int, double>();

            for (var networkIndex = 0; networkIndex < numNetworks; networkIndex++)
            {
                networkScoreDict[networkIndex] = 0;

                for (var iteration = 0; iteration < numIter; iteration++)
                {
                    var board = GetBoard(boardSize);
                    var distanceBefore = board.Distance();

                    var bestPossibleDistanceDiff = board.BestDistanceChangeInOneMove();

                    var output = networks[networkIndex].Propagate(board.GetPositions());
                    var direction = TranslateOutput(output);
                    board.Move(direction);

                    var distanceAfter = board.Distance();

                    networkScoreDict[networkIndex] += (distanceBefore - distanceAfter) / bestPossibleDistanceDiff / numIter;
                }
            }

            var orderedScores = networkScoreDict.OrderByDescending(s => s.Value).ToList();
            var bestScores = orderedScores.Take((int)(numNetworks * 0.75)).ToList();
            var bestNetworkIndexes = bestScores.Select(s => s.Key).ToList();
            var otherNetworkIndexes = Enumerable.Range(0, numNetworks).Except(bestNetworkIndexes).ToList();

            var bestNetworks = networks.Where(((_, index) => !otherNetworkIndexes.Contains(index))).ToList();
            var networksToSwap = networks.Where((_, index) => otherNetworkIndexes.Contains(index)).ToList();

            foreach (var network in networksToSwap)
            {
                networks.Remove(network);

                var goodNetworkCopy = CopyRandomGoodNetwork(bestNetworks);

                goodNetworkCopy.Mutate();

                networks.Add(goodNetworkCopy);
            }

            if (generation % 100 == 0)
            {
                Console.Clear();
                Console.WriteLine($"""
                                   Generation {generation}
                                   Best score: {bestScores[0].Value:0.00} (network {bestScores[0].Key})
                                   Average score: {networkScoreDict.Average(n => n.Value):0.00}
                                   Nth Generation time: {stopwatch.Elapsed.TotalSeconds:0.00} seconds
                                   """);
                
                stopwatch.Restart();
            }

            generation++;
        }
    }

    private static Network CopyRandomGoodNetwork(List<Network> networks)
    {
        var randomIndex = Random.Shared.Next(0, networks.Count);

        var networkToCopy = networks[randomIndex];

        return networkToCopy.Copy();
    }

    public static Board GetBoard(int size)
    {
        var randomStart = (Random.Shared.Next(0, size), Random.Shared.Next(0, size));
        var randomTarget = (Random.Shared.Next(0, size), Random.Shared.Next(0, size));

        if (randomStart == randomTarget)
        {
            if (randomTarget.Item1 < size)
                randomTarget.Item1++;
            else
                randomTarget.Item1--;
        }

        return new(size, size, randomTarget, randomStart);
    }

    public static Direction TranslateOutput(double[] output)
    {
        var o1 = output[0];
        var o2 = output[1];
        var o3 = output[2];
        var o4 = output[3];

        const double threshold = 0.9d;

        if (o1 >= threshold) return Direction.Up;
        if (o2 >= threshold) return Direction.Down;
        if (o3 >= threshold) return Direction.Left;
        if (o4 >= threshold) return Direction.Right;

        return Direction.Up;
    }

    public static double Sigmoid(double value)
    {
        var k = (double) Math.Exp((double) value);
        return k / (1.0d + k);
    }
}