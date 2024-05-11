namespace ReinforcementLearning;

public class Program
{
    private static readonly (int, int) BoardSize = (100, 100);

    private static async Task Main()
    {
        const int trainingIterations = 1000;
        const double initialLearningRate = 100d;
        const double learningRateDecay = 0.1d;

        var network = new Network([4, 4, 4], Activation.Sigmoid);

        var iteration = 1;

        while (true)
        {
            var losses = new List<double>();
            var learningRate = initialLearningRate / (1 + learningRateDecay * iteration);

            for (var i = 0; i < trainingIterations; i++)
            {
                network.Learn(() =>
                {
                    var board = Board.Generate(BoardSize);

                    var normalizedInputs = board.GetNormalizedPositions();

                    var output = network.Propagate(normalizedInputs);

                    var softmaxOutput = Activation.Softmax(output);

                    var direction = TranslateOutput(softmaxOutput);

                    var optimalReductionInDistance = board.OptimalReductionInDistance();

                    var preMoveDistance = board.Distance();

                    board.Move(direction);

                    var postMoveDistance = board.Distance();

                    board.UndoMove(direction);

                    var reductionInDistance = preMoveDistance - postMoveDistance;

                    var reductionRelativeToOptimal = reductionInDistance / optimalReductionInDistance;

                    var loss = 1 - reductionRelativeToOptimal;

                    losses.Add(loss);

                    return loss;

                }, learningRate);
            }

            Console.WriteLine($"Average loss for iteration {iteration}:\t {losses.Average()}\t-\tLR: {learningRate}");

            iteration++;
        }
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


}