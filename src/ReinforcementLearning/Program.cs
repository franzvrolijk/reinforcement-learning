using System.Collections.Concurrent;
using System.Diagnostics;

namespace ReinforcementLearning;

public class Program
{
    private static async Task Main()
    {
        var numNetworks = 40;
        var boardSize = 50;
        var numIter = 20;

        var networks = Enumerable.Range(0, numNetworks)
            .Select(_ => new Network([4, 8, 8, 4], Sigmoid))
            .ToList();

        var stopwatch = new Stopwatch();
        stopwatch.Start();

        var generation = 1;

        while (true)
        {
            var networkScoreDict = new ConcurrentDictionary<int, double>();

            var tasks = new List<Task>();

            for (var networkIndex = 0; networkIndex < numNetworks; networkIndex++)
            {
                var index = networkIndex;
                // TODO - reuse board data per network...?
                tasks.Add(Task.Run(() =>
                {
                    networkScoreDict[index] = 0;

                    for (var iteration = 0; iteration < numIter; iteration++)
                    {
                        var board = GetBoard(boardSize);
                        var distanceBefore = board.Distance();
                        var bestPossibleDistanceDiff = board.BestDistanceChangeInOneMove();

                        var normalizedInputs = board.GetPositions()
                            .Select(x => (double)x / (boardSize - 1))
                            .ToArray();

                        var rawOutput = networks[index].Propagate(normalizedInputs);

                        var output = Softmax(rawOutput);

                        var direction = TranslateOutput(output);

                        board.Move(direction);

                        var distanceAfter = board.Distance();
                        var decreaseInDistance = distanceBefore - distanceAfter;
                        var improvementRelativeToOptimal = decreaseInDistance / bestPossibleDistanceDiff;

                        networkScoreDict[index] += Math.Max(0, improvementRelativeToOptimal);
                    }
                }));
            }

            await Task.WhenAll(tasks);

            foreach (var (index, score) in networkScoreDict)
            {
                networkScoreDict[index] = score / numIter;
            }

            var orderedScores = networkScoreDict.OrderByDescending(s => s.Value).ToList();
            var bestScores = orderedScores.Take(numNetworks / 2).ToList();
            var bestNetworkIndexes = bestScores.Select(s => s.Key).ToList();

            var bestNetworks = networks.Where((_, index) => bestNetworkIndexes.Contains(index)).ToList();
            var networksToSwap = networks.Where((_, index) => !bestNetworkIndexes.Contains(index)).ToList();

            var averageScore = orderedScores.Average(s => s.Value);
            var probabilityOfMutation = 0.00001d;//Math.Clamp(1 - averageScore, 0, 1);

            foreach (var network in networksToSwap)
            {
                networks.Remove(network);

                var goodNetworkCopy = CopyRandomGoodNetwork(bestNetworks);

                // TODO - introduce gradient descent here
                goodNetworkCopy.MutateRandomly(probabilityOfMutation);

                networks.Add(goodNetworkCopy);
            }

            if (generation % 1000 == 0)
            {
                Console.Clear();
                Console.WriteLine($"""
                                   Generation {generation}
                                   Best score: {bestScores[0].Value:0.00} (network {bestScores[0].Key})
                                   Average score: {averageScore:0.00}
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
            if (randomTarget.Item1 < size / 2) randomTarget.Item1++;
            else randomTarget.Item1--;
        }

        return new(size, size, randomTarget, randomStart);
    }

    public static Direction TranslateOutput(double[] output)
    {
        var indexOfMax = 0;

        for (var i = 1; i < output.Length; i++)
        {
            if (output[i] > output[indexOfMax]) indexOfMax = i;
        }

        return indexOfMax switch
        {
            0 => Direction.Up,
            1 => Direction.Down,
            2 => Direction.Left,
            3 => Direction.Right,
            _ => throw new()
        };
    }

    public static double Sigmoid(double value)
    {
        var k = Math.Exp(value);
        return k / (1.0d + k);
    }


    public static double[] Softmax(double[] values)
    {
        var max = values.Max();
        var scale = 0.0;

        for (var i = 0; i < values.Length; i++)
        {
            values[i] = Math.Exp(values[i] - max);
            scale += values[i];
        }

        for (var i = 0; i < values.Length; i++)
        {
            values[i] /= scale;
        }

        return values;
    }
}