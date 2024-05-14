using ILGPU;
using System.Diagnostics;

namespace ReinforcementLearning;

public class Program
{
    private static readonly (int, int) BoardSize = (10, 10);

    private static void Main()
    {
        const int trainingIterations = 100;
        const double initialLearningRate = 0.2d;
        const double learningRateDecay = 0.1d;
        const double delta = 0.00000001d;

        var network = new Network([4, 8, 16, 4, 1], Activation.Sigmoid);
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

                    return loss;

                }, learningRate, delta, accelerator);
            }

            s.Stop();

            List<double> scores = [];

            for (var i = 0; i < 10; i++)
            {
                board = Board.Generate(BoardSize);

                normalizedInputs = board.GetNormalizedPositions();

                var direction = network.Propagate(normalizedInputs)[0];

                optimalReductionInDistance = board.OptimalReductionInDistance();

                preMoveDistance = board.Distance();

                board.Move(direction);

                var postMoveDistance = board.Distance();

                var reductionInDistance = preMoveDistance - postMoveDistance;

                var reductionRelativeToOptimal = reductionInDistance / optimalReductionInDistance;

                scores.Add(reductionRelativeToOptimal);
            }

            var min = scores.Min();
            var max = scores.Max();
            var (avg, spread) = (scores.Average(), max - min);
            Console.WriteLine($"(Iteration {iteration})\tAvg: {avg:0.00}\tMin: {min:0.00}\tMax {max:0.00}\tSpread: {spread:0.00}\tLR: {learningRate:0.00}\tTime: {s.ElapsedMilliseconds}ms");

            iteration++;
        }
    }
}