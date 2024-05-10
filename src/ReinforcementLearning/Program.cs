namespace ReinforcementLearning;

public class Program
{
    private static readonly (int, int) BoardSize = (100, 100);

    private static async Task Main()
    {
        var network = new Network([4, 8, 8, 4], Activation.Sigmoid);

        var trainingIterations = 1000;
        var measureIterations = 100;
        var measureBoards = Enumerable.Range(0, measureIterations).Select(_ => Board.Generate(BoardSize)).ToArray();

        var iteration = 1;

        var previousAvgLoss = 1d;
        while (true)
        {
            // Train
            for (var i = 0; i < trainingIterations; i++)
            {
                var board = Board.Generate(BoardSize);

                var normalizedInputs = board.GetNormalizedPositions();

                var loss = () =>
                {
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

                    return 1 - reductionRelativeToOptimal;
                };

                var learningRate = 1 * previousAvgLoss;

                network.Learn(loss, learningRate);
            }

            // Measure performance
            var totalLoss = 0d;

            for (var i = 0; i < measureIterations; i++)
            {
                var board = measureBoards[i];

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

                totalLoss += loss;
            }

            var averageLoss = totalLoss / measureIterations;
            previousAvgLoss = averageLoss;
            
            Console.Clear();
            Console.WriteLine($"#{iteration} - Average loss: {averageLoss}");

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