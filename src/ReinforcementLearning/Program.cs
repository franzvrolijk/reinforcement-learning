using Timer = System.Timers.Timer;

namespace ReinforcementLearning;

public class Program
{
    private static int _frameCount = 0;
    private static int _framerate;

    private static void Main()
    {
        new Network([4, 4, 3], _ => _);
        return;
        var board = new Board(16, 9);
        var cancellationToken = new CancellationToken();

        var timer = new Timer
        {
            Interval = 1000
        };

        timer.Elapsed += (_, _) =>
        {
            lock (board)
            {
                _framerate = _frameCount;
                _frameCount = 0;
            }
        };

        timer.Start();


        while (!cancellationToken.IsCancellationRequested)
        {
            Console.Clear();
            Console.WriteLine($"-----{_framerate} FPS-----");
            board.Print();

            lock (board)
            {
                _frameCount++;
            }

            Thread.Sleep(100);
        }

        timer.Stop();
    }
}

public enum Direction
{
    Up, Down, Left, Right
}

public class Network
{
    // [layer][node]
    private Node[][] _nodes;

    // [fromLayer][fromNode][toNode]
    private decimal[][][] _weights;

    private int[] _layerSizes;

    private readonly Func<decimal, decimal> _activationFunction;

    public Network(int[] layerSizes, Func<decimal, decimal> activationFunction, decimal[][][]? weights = null)
    {
        _layerSizes = layerSizes;
        _activationFunction = activationFunction;

        _nodes = layerSizes.Select(i => Enumerable.Range(0, i).Select(_ => new Node()).ToArray()).ToArray();

        if (weights != null)
        {
            _weights = weights;
            return;
        }

        _weights = new decimal[layerSizes.Length - 1][][];

        // Generate random weights
        for (var i = 0; i < layerSizes.Length - 1; i++)
        {
            // Weights from layer i to layer i + 1
            var currentLayerSize = layerSizes[i];
            var nextLayerSize = layerSizes[i + 1];


            _weights[i] = Enumerable.Range(0, currentLayerSize)
                .Select(_ => Enumerable.Range(0, nextLayerSize)
                    .Select(_ => (decimal)Random.Shared.NextDouble())
                    .ToArray())
                .ToArray();
        }
    }

    private decimal Activation(decimal x) => _activationFunction(x);

    public decimal[] Propagate(int[] inputs)
    {
        // Feed inputs to input layer
        for (var nodeIndex = 0; nodeIndex < inputs.Length; nodeIndex++)
        {
            _nodes[0][nodeIndex].Value = inputs[nodeIndex];
        }

        // For every layer except first and last
        for (var layerIndex = 0; layerIndex < _layerSizes.Length - 1; layerIndex++)
        {
            var currentLayer = _nodes[layerIndex];
            var nextLayer = _nodes[layerIndex + 1];

            // For every combination of node in current and next layer
            for (var currentLayerNodeIndex = 0; currentLayerNodeIndex < currentLayer.Length; currentLayerNodeIndex++)
            {
                for (var nextLayerNodeIndex = 0; nextLayerNodeIndex < nextLayer.Length; nextLayerNodeIndex++)
                {
                    // Find weight and calculate next layer node's value
                    var weight = _weights[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex];
                    nextLayer[nextLayerNodeIndex].Value += currentLayer[currentLayerNodeIndex].Value * weight;
                }
            }

            foreach (var node in nextLayer)
            {
                var preActivationValue = node.Value;
                node.Value = Activation(preActivationValue);
            }
        }

        // Calculate outputs
        return _nodes
            .Last()
            .Select(node => node.Value)
            .ToArray();
    }



    class Node
    {
        public decimal Value { get; set; }
    }
}

public class Board
{
    private readonly char[][] _board;
    private readonly int _width;
    private readonly int _height;

    public Board(int width, int height)
    {
        _width = width;
        _height = height;
        _board = new char[width][];

        for (var i = 0; i < width; i++)
        {
            _board[i] = new char[height];
        }

        for (var y = 0; y < _height; y++)
        {
            _board[0][y] = 'x';
            for (var x = 1; x < _width; x++)
            {
                _board[x][y] = $"{y}".ToCharArray()[0];
            }
        }
    }

    public char[] this[int i] => _board[i];

    public void Print()
    {
        for (var y = 0; y < _height; y++)
        {
            var y1 = y;
            var row = string.Concat(_board[.._width].Select(column => $"{column[y1]} "));

            Console.WriteLine(row);
        }
    }
}