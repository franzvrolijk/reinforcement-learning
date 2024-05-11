namespace ReinforcementLearning;

public class Network
{
    // _nodes[layerIndex][nodeIndex]
    private readonly double[][] _nodes;

    // _weights[fromLayerIndex][fromNodeIndex][toNodeIndex]
    private readonly double[][][] _weights;

    private readonly double[][][] _biases;

    private readonly double[][][] _weightGradients;

    private readonly double[][][] _biasGradients;

    private readonly int[] _layerSizes;

    private readonly Func<double, double> _activationFunction;

    public Network(int[] layerSizes, Func<double, double> activationFunction, double[][][]? weights = null, double[][][]? biases = null)
    {
        _layerSizes = layerSizes;
        _activationFunction = activationFunction;
        
        _nodes = layerSizes.Select(i => Enumerable.Range(0, i).Select(_ => 0d).ToArray()).ToArray();

        if (weights != null)
        {
            _weights = weights;
        }
        else
        {
            _weights = new double[layerSizes.Length - 1][][];

            // Generate weights
            for (var i = 0; i < layerSizes.Length - 1; i++)
            {
                // Weights from layer i to layer i + 1
                var currentLayerSize = layerSizes[i];
                var nextLayerSize = layerSizes[i + 1];


                _weights[i] = Enumerable.Range(0, currentLayerSize)
                    .Select(_ => Enumerable.Range(0, nextLayerSize)
                        .Select(_ => 0.01 * Random.Shared.NextDouble() - 0.005)
                        .ToArray())
                    .ToArray();
            }
        }

        if (biases != null)
        {
            _biases = biases;
        }
        else
        {
            _biases = new double[layerSizes.Length - 1][][];

            // Generate biases
            for (var i = 0; i < layerSizes.Length - 1; i++)
            {
                // Weights from layer i to layer i + 1
                var currentLayerSize = layerSizes[i];
                var nextLayerSize = layerSizes[i + 1];


                _biases[i] = Enumerable.Range(0, currentLayerSize)
                    .Select(_ => Enumerable.Range(0, nextLayerSize)
                        .Select(_ => 0.01)
                        .ToArray())
                    .ToArray();
            }
        }
    }

    public Network Copy()
    {
        return new (_layerSizes, _activationFunction, _weights.Select(w => w).ToArray(), _biases.Select(b => b).ToArray());
    }

    private double Activation(double x) => _activationFunction(x);

    public double[] Propagate(double[] inputs)
    {
        // Feed inputs to input layer
        for (var nodeIndex = 0; nodeIndex < inputs.Length; nodeIndex++)
        {
            _nodes[0][nodeIndex] = inputs[nodeIndex];
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
                    // Find weight/bias and calculate next layer node's value
                    var weight = _weights[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex];
                    var bias = _biases[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex];
                    nextLayer[nextLayerNodeIndex] += (currentLayer[currentLayerNodeIndex] * weight) + bias;
                }
            }

            // Run activation function on each node in next layer
            for (var nextLayerNodeIndex = 0; nextLayerNodeIndex < nextLayer.Length; nextLayerNodeIndex++)
            {
                nextLayer[nextLayerNodeIndex] = Activation(nextLayer[nextLayerNodeIndex]);
            }
        }

        // Calculate outputs
        return [.. _nodes.Last()];
    }

    public void Learn(Func<double> loss, double learnRate)
    {
        const double x = 0.00000001d;

        var weightGradients = new Dictionary<(int, int, int), double>();
        var biasGradients = new Dictionary<(int, int, int), double>();

        // Compute gradients of loss function based on change in weights/biases
        for (var layerIndex = 0; layerIndex < _layerSizes.Length - 1; layerIndex++)
        {
            var currentLayerSize = _layerSizes[layerIndex];
            var nextLayerSize = _layerSizes[layerIndex + 1];

            for (var currentLayerNodeIndex = 0; currentLayerNodeIndex < currentLayerSize; currentLayerNodeIndex++)
            {
                for (var nextLayerNodeIndex = 0; nextLayerNodeIndex < nextLayerSize; nextLayerNodeIndex++)
                {
                    var lossBefore = loss();

                    _weights[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex] += x;

                    var lossAfter = loss();

                    var deltaLoss = lossAfter - lossBefore;

                    var derivative = deltaLoss / x;

                    weightGradients[(layerIndex, currentLayerNodeIndex, nextLayerNodeIndex)] = derivative;

                    lossBefore = lossAfter;

                    _biases[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex] += x;

                    lossAfter = loss();

                    deltaLoss = lossAfter - lossBefore;

                    derivative = deltaLoss / x;

                    biasGradients[(layerIndex, currentLayerNodeIndex, nextLayerNodeIndex)] = derivative;

                    _weights[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex] -= x;
                    _biases[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex] -= x;
                }
            }
        }

        // Update weights and biases based on gradients
        for (var layerIndex = 0; layerIndex < _layerSizes.Length - 1; layerIndex++)
        {
            var currentLayerSize = _layerSizes[layerIndex];
            var nextLayerSize = _layerSizes[layerIndex + 1];

            for (var currentLayerNodeIndex = 0; currentLayerNodeIndex < currentLayerSize; currentLayerNodeIndex++)
            {
                for (var nextLayerNodeIndex = 0; nextLayerNodeIndex < nextLayerSize; nextLayerNodeIndex++)
                {
                    var (weightGradient, biasGradient) = (weightGradients[(layerIndex, currentLayerNodeIndex, nextLayerNodeIndex)], biasGradients[(layerIndex, currentLayerNodeIndex, nextLayerNodeIndex)]);

                    var weightAddition = weightGradient < 0 
                        ? learnRate 
                        : learnRate * -1;

                    var biasAddition = biasGradient < 0
                        ? learnRate
                        : learnRate * -1;

                    _weights[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex] += weightAddition;
                    _biases[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex] += biasAddition;
                }
            }
        }
    }
}