namespace ReinforcementLearning;

public class Network
{
    // _nodes[layerIndex][nodeIndex]
    private readonly double[][] _nodes;

    // Weights[fromLayerIndex][fromNodeIndex][toNodeIndex]
    public readonly double[][][] Weights;

    public readonly double[][][] Biases;

    private readonly int[] _layerSizes;

    private readonly Func<double, double> _activationFunction;
    private static double RandomWeightValue => 0.01 * Random.Shared.NextDouble() - 0.005;

    public Network(int[] layerSizes, Func<double, double> activationFunction, double[][][]? weights = null, double[][][]? biases = null)
    {
        _layerSizes = layerSizes;
        _activationFunction = activationFunction;
        
        _nodes = layerSizes.Select(i => Enumerable.Range(0, i).Select(_ => 0d).ToArray()).ToArray();

        if (weights != null)
        {
            Weights = weights;
        }
        else
        {
            Weights = new double[layerSizes.Length - 1][][];

            // Generate weights
            for (var i = 0; i < layerSizes.Length - 1; i++)
            {
                // Weights from layer i to layer i + 1
                var currentLayerSize = layerSizes[i];
                var nextLayerSize = layerSizes[i + 1];


                Weights[i] = Enumerable.Range(0, currentLayerSize)
                    .Select(_ => Enumerable.Range(0, nextLayerSize)
                        .Select(_ => RandomWeightValue)
                        .ToArray())
                    .ToArray();
            }
        }

        if (biases != null)
        {
            Biases = biases;
        }
        else
        {
            Biases = new double[layerSizes.Length - 1][][];

            // Generate biases
            for (var i = 0; i < layerSizes.Length - 1; i++)
            {
                // Weights from layer i to layer i + 1
                var currentLayerSize = layerSizes[i];
                var nextLayerSize = layerSizes[i + 1];


                Biases[i] = Enumerable.Range(0, currentLayerSize)
                    .Select(_ => Enumerable.Range(0, nextLayerSize)
                        .Select(_ => 0.01)
                        .ToArray())
                    .ToArray();
            }
        }
    }

    public Network Copy()
    {
        return new (_layerSizes, _activationFunction, Weights.Select(w => w).ToArray(), Biases.Select(b => b).ToArray());
    }

    private double Activation(double x) => _activationFunction(x);

    public double[] Propagate(double[] inputs)
    {
        // Feed inputs to input layer
        for (var nodeIndex = 0; nodeIndex < inputs.Length; nodeIndex++)
        {
            _nodes[0][nodeIndex] = inputs[nodeIndex];
        }

        // For every layer except last
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
                    var weight = Weights[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex];
                    var bias = Biases[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex];
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

    public void GradientDescent(Func<double> loss, double learnRate, double delta)
    {
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

                    Weights[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex] += delta;

                    var lossAfter = loss();

                    var deltaLoss = lossAfter - lossBefore;

                    var derivative = deltaLoss / delta;

                    weightGradients[(layerIndex, currentLayerNodeIndex, nextLayerNodeIndex)] = derivative;

                    Weights[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex] -= delta;

                    lossBefore = lossAfter;

                    Biases[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex] += delta;

                    lossAfter = loss();

                    deltaLoss = lossAfter - lossBefore;

                    derivative = deltaLoss / delta;

                    biasGradients[(layerIndex, currentLayerNodeIndex, nextLayerNodeIndex)] = derivative;

                    Biases[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex] -= delta;
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

                    Weights[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex] += weightAddition;
                    Biases[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex] += biasAddition;
                }
            }
        }
    }
}