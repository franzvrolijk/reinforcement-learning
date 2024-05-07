namespace ReinforcementLearning;

public class Network
{
    // _nodes[layerIndex][nodeIndex]
    private readonly Node[][] _nodes;

    // _weights[fromLayerIndex][fromNodeIndex][toNodeIndex]
    private readonly double[][][] _weights;

    private readonly int[] _layerSizes;

    private readonly Func<double, double> _activationFunction;

    public Network(int[] layerSizes, Func<double, double> activationFunction, double[][][]? weights = null)
    {
        _layerSizes = layerSizes;
        _activationFunction = activationFunction;

        _nodes = layerSizes.Select(i => Enumerable.Range(0, i).Select(_ => new Node()).ToArray()).ToArray();

        if (weights != null)
        {
            _weights = weights;
            return;
        }

        _weights = new double[layerSizes.Length - 1][][];

        // Generate random weights
        for (var i = 0; i < layerSizes.Length - 1; i++)
        {
            // Weights from layer i to layer i + 1
            var currentLayerSize = layerSizes[i];
            var nextLayerSize = layerSizes[i + 1];


            _weights[i] = Enumerable.Range(0, currentLayerSize)
                .Select(_ => Enumerable.Range(0, nextLayerSize)
                    .Select(_ => (double)Random.Shared.NextDouble())
                    .ToArray())
                .ToArray();
        }
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
                    if (Random.Shared.NextDouble() >= 0.8)
                        continue;

                    _weights[layerIndex][currentLayerNodeIndex][nextLayerNodeIndex] = Random.Shared.NextDouble();
                }
            }
        }
    }


    private class Node
    {
        public double Value { get; set; }
    }
}