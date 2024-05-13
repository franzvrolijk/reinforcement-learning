
namespace ReinforcementLearning;

public class Board
{
    private readonly int _width;
    private readonly int _height;

    private (double, double) _targetPos;
    private (double, double) _currentPos;

    public Board(int width, int height, (double, double) targetPos, (double, double) startPos)
    {
        _targetPos = targetPos;
        _currentPos = startPos;
        _width = width;
        _height = height;
    }

    public static Board Generate((int, int) size)
    {
        var (width, height) = size;

        var randomStart = (Random.Shared.NextDouble() * width, Random.Shared.NextDouble() * height);
        var randomTarget = (Random.Shared.NextDouble() * width, Random.Shared.NextDouble() * height);

        while (EuclidianDistance(randomStart, randomTarget) <= 1)
        {
            randomTarget = (Random.Shared.NextDouble() * width, Random.Shared.NextDouble() * height);
        }

        return new(width, height, randomTarget, randomStart);
    }

    public double[] GetPositions()
    {
        return [_targetPos.Item1, _targetPos.Item2, _currentPos.Item1, _currentPos.Item2];
    }

    public double[] GetNormalizedPositions()
    {
        return
        [
            _targetPos.Item1 / _width,
            _targetPos.Item2 / _height,
            _currentPos.Item1 / _width,
            _currentPos.Item2 / _height
        ];
    }

    public double Distance() => EuclidianDistance(_targetPos, _currentPos);

    private static double EuclidianDistance((double, double) targetPos, (double, double) currentPos)
    {
        var diffX = targetPos.Item1 - currentPos.Item1;
        var diffY = targetPos.Item2 - currentPos.Item2;

        return Math.Sqrt(diffX * diffX + diffY * diffY);
    }

    public double OptimalReductionInDistance()
    {
        // TODO - smart logic about overshooting
        return 1;
    }

    public void Move(double directionAsFloat)
    {
        // Convert float to radians directly
        var directionAsRadians = directionAsFloat * 2 * Math.PI;

        // Calculate the new position
        var newX = _currentPos.Item1 + Math.Cos(directionAsRadians);
        var newY = _currentPos.Item2 + Math.Sin(directionAsRadians);

        // Update the current position
        _currentPos = (newX, newY);
    }

    public void UndoMove(double directionAsFloat)
    {
        var invertedDirection = directionAsFloat < 0.5
            ? directionAsFloat + 0.5
            : directionAsFloat - 0.5;

        Move(invertedDirection);
    }
}