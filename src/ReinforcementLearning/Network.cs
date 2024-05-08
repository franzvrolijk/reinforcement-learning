namespace ReinforcementLearning;

public class Network
{
    // _nodes[layerIndex][nodeIndex]
    private readonly Node[][] _nodes;

    // _weights[fromLayerIndex][fromNodeIndex][toNodeIndex]
    private readonly double[][][] _weights;

    private readonly double[][][] _biases;

    private readonly int[] _layerSizes;

    private readonly Func<double, double> _activationFunction;

    public Network(int[] layerSizes, Func<double, double> activationFunction, double[][][]? weights = null, double[][][]? biases = null)
    {
        _layerSizes = layerSizes;
        _activationFunction = activationFunction;

        _nodes = layerSizes.Select(i => Enumerable.Range(0, i).Select(_ => new Node()).ToArray()).ToArray();

        if (weights != null)
        {
            _weights = weights;
        }
        else
        {
            _weights = new double[layerSizes.Length - 1][][];

            // Generate random weights
            for (var i = 0; i < layerSizes.Length - 1; i++)
            {
                // Weights from layer i to layer i + 1
                var currentLayerSize = layerSizes[i];
                var nextLayerSize = layerSizes[i + 1];


                _weights[i] = Enumerable.Range(0, currentLayerSize)
                    .Select(_ => Enumerable.Range(0, nextLayerSize)
                        .Select(_ => (Random.Shared.NextDouble() * 2) - 1)
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

            // Generate random biases
            for (var i = 0; i < layerSizes.Length - 1; i++)
            {
                // Weights from layer i to layer i + 1
                var currentLayerSize = layerSizes[i];
                var nextLayerSize = layerSizes[i + 1];


                _biases[i] = Enumerable.Range(0, currentLayerSize)
                    .Select(_ => Enumerable.Range(0, nextLayerSize)
                        .Select(_ => (Random.Shared.NextDouble() * 2) - 1)
                        .ToArray())
                    .ToArray();
            }
        }
    }

    public Network Copy()
    {
        return new Network(_layerSizes, _activationFunction, _weights.Select(w => w).ToArray(), _biases.Select(b => b).ToArray());
    }

    private double Activation(double x) => _activationFunction(x);

    public double[] Propagate(int[] inputs)
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
                    var bias = _biases[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex];
                    nextLayer[nextLayerNodeIndex].Value += (currentLayer[currentLayerNodeIndex].Value * weight) + bias;
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

    public void Mutate()
    {
        for (var layerIndex = 0; layerIndex < _layerSizes.Length - 1; layerIndex++)
        {
            var currentLayerSize = _layerSizes[layerIndex];
            var nextLayerSize = _layerSizes[layerIndex + 1];

            for (var currentLayerNodeIndex = 0; currentLayerNodeIndex < currentLayerSize; currentLayerNodeIndex++)
            {
                for (var nextLayerNodeIndex = 0; nextLayerNodeIndex < nextLayerSize; nextLayerNodeIndex++)
                {
                    if (Random.Shared.NextDouble() > 0.95)
                    {
                        var current = _weights[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex];

                        _weights[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex] = Math.Clamp(current * ((2 * Random.Shared.NextDouble()) - 1), -1, 1);
                    }

                    if (Random.Shared.NextDouble() > 0.95)
                    {
                        var current = _biases[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex];
                        
                        _biases[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex] = Math.Clamp(current * ((2 * Random.Shared.NextDouble()) - 1), -1, 1);
                    }
                    
                }
            }
        }
    }

    private class Node
    {
        public double Value { get; set; }
    }
}