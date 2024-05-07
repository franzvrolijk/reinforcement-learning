using System.Diagnostics;

namespace ReinforcementLearning;

public class Program
{
    private static async Task Main()
    {
        var numNetworks = 20;
        var boardSize = 10;
        var numIter = 3;
        var generation = 1;

        var networks = Enumerable.Range(0, numNetworks)
            .Select(_ => new Network([4, 8, 4, 2], Sigmoid))
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

            var bestScores = networkScoreDict.OrderByDescending(s => s.Value).Take(numNetworks / 3).ToList();
            var bestNetworkIndexes = bestScores.Select(s => s.Key).ToList();
            var otherNetworkIndexes = Enumerable.Range(0, numNetworks).Except(bestNetworkIndexes).ToList();
            var networksToMutate = networks.Where((_, i) => otherNetworkIndexes.Contains(i)).ToList();

            // Pass param to mutate that will change mutation aggresiveness? "score relative to best network (%)"
            networksToMutate.ForEach(n => n.Mutate());

            if (generation % 10000 == 0)
            {
                Console.WriteLine($"""
                                   Generation {generation}
                                   Best score: {bestScores[0].Value} (network {bestScores[0].Key})
                                   Average score: {networkScoreDict.Average(n => n.Value)}
                                   Nth Generation time: {stopwatch.Elapsed.TotalSeconds} seconds
                                   """);
                Console.WriteLine();
                stopwatch.Restart();
            }

            generation++;
        }
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

        const double threshold = 0.5d;

        return o1 switch
        {
            < threshold when o2 >= threshold => Direction.Up,
            >= threshold when o2 < threshold => Direction.Down,
            < threshold when o2 < threshold => Direction.Left,
            _ => Direction.Right
        };
    }

    public static double Sigmoid(double value)
    {
        var k = (double) Math.Exp((double) value);
        return k / (1.0d + k);
    }
}