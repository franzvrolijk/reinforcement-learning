using ILGPU;
using System.Diagnostics;

namespace ReinforcementLearning;

public class Program
{
    private static readonly (int, int) BoardSize = (10, 10);

    private static void Main()
    {
        const int trainingIterations = 1000;
        const double initialLearningRate = 0.001d;
        const double learningRateDecay = 0.001d;
        const double delta = 0.00000001d;

        var testData = Enumerable.Range(0, 10).Select(_ =>
        {
            var board = Board.Generate(BoardSize);

            return new
            {
                Board = board,
                PreMoveDistance = board.Distance(),
                OptimalReductionInDistance = board.OptimalReductionInDistance(),
                NormalizedPositions = board.GetNormalizedPositions()
            };
        }).ToList();

        var network = new Network([4, 8, 4, 1], Activation.Sigmoid);
        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(false).CreateAccelerator(context);

        var iteration = 1;

        while (true)
        {
            var learningRate = initialLearningRate / (1 + learningRateDecay * iteration);
            
            var board = Board.Generate(BoardSize);
            var normalizedInputs = board.GetNormalizedPositions();
            var optimalReductionInDistance = board.OptimalReductionInDistance();
            var preMoveDistance = board.Distance();

            var s = Stopwatch.StartNew();

            for (var i = 0; i < trainingIterations; i++)
            {
                network.GradientDescentGpu(() =>
                {
                    var direction = network.Propagate(normalizedInputs)[0];

                    var postMoveDistance = board.DistanceAfterMove(direction);

                    var reductionInDistance = preMoveDistance - postMoveDistance;

                    var reductionRelativeToOptimal = reductionInDistance / optimalReductionInDistance;

                    var loss = 1 - reductionRelativeToOptimal;

                    return loss * loss;

                }, learningRate, delta, accelerator);
            }

            s.Stop();

            List<double> scores = [];

            for (var i = 0; i < 10; i++)
            {
                var testCase = testData[i];

                var direction = network.Propagate(testCase.NormalizedPositions)[0];

                var postMoveDistance = testCase.Board.DistanceAfterMove(direction);

                var reductionInDistance = testCase.PreMoveDistance - postMoveDistance;

                var reductionRelativeToOptimal = reductionInDistance / testCase.OptimalReductionInDistance;

                scores.Add(reductionRelativeToOptimal);
            }

            var min = scores.Min();
            var max = scores.Max();
            var (avg, spread) = (scores.Average(), max - min);

            if (Console.KeyAvailable)
            {
                Console.ReadKey(true);
                Console.Clear();
                network.Print();
                Console.WriteLine($"(Iteration {iteration})\tAvg: {avg:0.00}\tMin: {min:0.00}\tMax {max:0.00}\tSpread: {spread:0.00}\tLR: {learningRate:0.000000}\tTime: {s.ElapsedMilliseconds}ms");
            }

            iteration++;
        }
    }
}