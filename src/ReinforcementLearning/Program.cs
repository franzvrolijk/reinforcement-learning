namespace ReinforcementLearning;

public class Program
{
    private static readonly (int, int) BoardSize = (100, 100);

    private static void Main()
    {
        const int trainingIterations = 1000;
        const double initialLearningRate = 0.5d;
        const double learningRateDecay = 0.01d;
        const double delta = 0.00000001d;

        var network = new Network([4, 8, 16, 4, 1], Activation.Sigmoid);

        var iteration = 1;

        while (true)
        {
            var learningRate = initialLearningRate / (1 + learningRateDecay * iteration);

            for (var i = 0; i < trainingIterations; i++)
            {
                network.Learn(() =>
                {
                    var board = Board.Generate(BoardSize);

                    var normalizedInputs = board.GetNormalizedPositions();

                    var direction = network.Propagate(normalizedInputs)[0];

                    var optimalReductionInDistance = board.OptimalReductionInDistance();

                    var preMoveDistance = board.Distance();

                    board.Move(direction);

                    var postMoveDistance = board.Distance();

                    var reductionInDistance = preMoveDistance - postMoveDistance;

                    var reductionRelativeToOptimal = reductionInDistance / optimalReductionInDistance;

                    var loss = 1 - reductionRelativeToOptimal;

                    return loss;

                }, learningRate, delta);
            }

            List<double> scores = [];

            for (var i = 0; i < 10; i++)
            {
                var board = Board.Generate(BoardSize);

                var normalizedInputs = board.GetNormalizedPositions();

                var direction = network.Propagate(normalizedInputs)[0];

                var optimalReductionInDistance = board.OptimalReductionInDistance();

                var preMoveDistance = board.Distance();

                board.Move(direction);

                var postMoveDistance = board.Distance();

                var reductionInDistance = preMoveDistance - postMoveDistance;

                var reductionRelativeToOptimal = reductionInDistance / optimalReductionInDistance;

                scores.Add(reductionRelativeToOptimal);
            }

            var min = scores.Min();
            var max = scores.Max();
            var (avg, spread) = (scores.Average(), max - min);
            Console.WriteLine($"(Iteration {iteration})\tAvg: {avg:0.00}\tMin: {min:0.00}\tMax {max:0.00}\tSpread: {spread:0.00}\tLR: {learningRate:0.00}");

            iteration++;
        }
    }
}