
namespace ReinforcementLearning;

public class Board
{
    private readonly int _width;
    private readonly int _height;

    private (int, int) _targetPos;
    private (int, int) _currentPos;

    public Board(int width, int height, (int, int) targetPos, (int, int) startPos)
    {
        _targetPos = targetPos;
        _currentPos = startPos;
        _width = width;
        _height = height;
    }

    public int[] GetPositions()
    {
        return [_targetPos.Item1, _targetPos.Item2, _currentPos.Item1, _currentPos.Item2];
    }

    public double Distance() => EuclidianDistance(_targetPos, _currentPos);

    private double EuclidianDistance((int, int) targetPos, (int, int) currentPos)
    {
        var diffX = targetPos.Item1 - currentPos.Item1;
        var diffY = targetPos.Item2 - currentPos.Item2;

        return Math.Sqrt(diffX * diffX + diffY * diffY);
    }

    public double BestDistanceChangeInOneMove()
    {
        var currentDistance = Distance();

        var d1 = EuclidianDistance(_targetPos, (_currentPos.Item1 + 1, _currentPos.Item2));
        var d2 = EuclidianDistance(_targetPos, (_currentPos.Item1 - 1, _currentPos.Item2));
        var d3 = EuclidianDistance(_targetPos, (_currentPos.Item1, _currentPos.Item2 + 1));
        var d4 = EuclidianDistance(_targetPos, (_currentPos.Item1, _currentPos.Item2 - 1));

        return new []
        {
            currentDistance - d1, 
            currentDistance - d2, 
            currentDistance - d3, 
            currentDistance - d4
        }.Min();
    }

    public void Move(Direction direction)
    {
        switch (direction)
        {
            case Direction.Up:
                _currentPos.Item2--;
                break;
            case Direction.Down:
                _currentPos.Item2++;
                break;
            case Direction.Left:
                _currentPos.Item1--;
                break;
            case Direction.Right:
                _currentPos.Item1++;
                break;
            default:
                throw new($"Invalid direction");
        }
    }

    public void Print()
    {
        for (var y = 0; y < _height; y++)
        {
            var row = Enumerable
                .Range(0, _width)
                .Select(x => _currentPos == (x, y) 
                    ? 'O' 
                    : _targetPos == (x, y) 
                        ? 'X' 
                        : '-')
                .ToArray();
            

            Console.WriteLine(row);
        }
    }
}

public enum Direction
{
    Up, Down, Left, Right
}