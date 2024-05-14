using ILGPU;
using ILGPU.Runtime;

namespace ReinforcementLearning;

public class Network
{
    private readonly double[] _nodes;

    public readonly double[] Weights;

    public readonly double[] Biases;

    private readonly int[] _layerSizes;

    private readonly Func<double, double> _activationFunction;
    private static double RandomWeightValue => 0.01 * Random.Shared.NextDouble() - 0.005;

    public Network(int[] layerSizes, Func<double, double> activationFunction, double[]? weights = null, double[]? biases = null)
    {
        _layerSizes = layerSizes;
        _activationFunction = activationFunction;

        _nodes = new double[_layerSizes.Sum()];

        var totalWeightsAndBiases = 0;

        for (var i = 0; i < layerSizes.Length - 1; i++)
        {
            totalWeightsAndBiases += layerSizes[i] * layerSizes[i + 1];
        }

        Weights = weights ?? Enumerable.Range(0, totalWeightsAndBiases).Select(_ => RandomWeightValue).ToArray();
        Biases = biases ?? Enumerable.Range(0, totalWeightsAndBiases).Select(_ => 0.01).ToArray();
    }

    public Network Copy()
    {
        return new(_layerSizes, _activationFunction, Weights.Select(w => w).ToArray(), Biases.Select(b => b).ToArray());
    }

    private double Activation(double x) => _activationFunction(x);

    public double[] Propagate(double[] inputs)
    {
        // Feed inputs to input layer
        for (var inputIndex = 0; inputIndex < inputs.Length; inputIndex++)
        {
            _nodes[inputIndex] = inputs[inputIndex];
        }

        // Find weight/bias and calculate next layer node's value
        var weightAndBiasIndex = 0;
        var nodeIndex = 0;
        for (var layerIndex = 0; layerIndex < _layerSizes.Length - 1; layerIndex++)
        {
            for (var currentNodeIndex = 0; currentNodeIndex < _layerSizes[layerIndex]; currentNodeIndex++)
            {
                for (var nextNodeIndex = 0; nextNodeIndex < _layerSizes[layerIndex + 1]; nextNodeIndex++)
                {
                    // Calculate the index of the next node in the _nodes array
                    var nextNodeArrayIndex = nodeIndex + _layerSizes[layerIndex] + nextNodeIndex;

                    // Run activation function on each node in next layer
                    _nodes[nextNodeArrayIndex] += Activation(_nodes[nodeIndex + currentNodeIndex] * Weights[weightAndBiasIndex] + Biases[weightAndBiasIndex]);

                    weightAndBiasIndex++;
                }
            }
            nodeIndex += _layerSizes[layerIndex];
        }

        var nodesBeforeOutput = _nodes.Length - _layerSizes[^1];

        // Calculate outputs
        return _nodes[nodesBeforeOutput..];
    }

    public void GradientDescent(Func<double> loss, double learnRate, double delta)
    {
        var weightGradients = new double[Weights.Length];
        var biasGradients = new double[Biases.Length];

        // Compute gradients of loss function based on change in weights/biases
        var weightAndBiasIndex = 0;
        for (var layerIndex = 0; layerIndex < _layerSizes.Length - 1; layerIndex++)
        {
            for (var currentLayerNodeIndex = 0; currentLayerNodeIndex < _layerSizes[layerIndex]; currentLayerNodeIndex++)
            {
                for (var nextLayerNodeIndex = 0; nextLayerNodeIndex < _layerSizes[layerIndex + 1]; nextLayerNodeIndex++)
                {
                    var lossBefore = loss();

                    Weights[weightAndBiasIndex] += delta;

                    var lossAfter = loss();

                    var deltaLoss = lossAfter - lossBefore;

                    var derivative = deltaLoss / delta;

                    weightGradients[weightAndBiasIndex] = derivative < 0
                        ? learnRate
                        : learnRate * -1; ;

                    Weights[weightAndBiasIndex] -= delta;

                    lossBefore = lossAfter;

                    Biases[weightAndBiasIndex] += delta;

                    lossAfter = loss();

                    deltaLoss = lossAfter - lossBefore;

                    derivative = deltaLoss / delta;

                    biasGradients[weightAndBiasIndex] = derivative < 0
                        ? learnRate
                        : learnRate * -1; ;

                    Biases[weightAndBiasIndex] -= delta;

                    weightAndBiasIndex++;
                }
            }
        }

        // Update weights and biases based on gradients
        weightAndBiasIndex = 0;
        for (var layerIndex = 0; layerIndex < _layerSizes.Length - 1; layerIndex++)
        {
            for (var currentLayerNodeIndex = 0; currentLayerNodeIndex < _layerSizes[layerIndex]; currentLayerNodeIndex++)
            {
                for (var nextLayerNodeIndex = 0; nextLayerNodeIndex < _layerSizes[layerIndex + 1]; nextLayerNodeIndex++)
                {
                    Weights[weightAndBiasIndex] += weightGradients[weightAndBiasIndex];
                    Biases[weightAndBiasIndex] += biasGradients[weightAndBiasIndex];

                    weightAndBiasIndex++;
                }
            }
        }
    }

    public void GradientDescentGpu(Func<double> loss, double learnRate, double delta, Accelerator accelerator)
    {
        var weightGradients = new double[Weights.Length];
        var biasGradients = new double[Biases.Length];

        var lossBefore = loss();

        // Compute gradients of loss function based on change in weights/biases
        var weightAndBiasIndex = 0;
        for (var layerIndex = 0; layerIndex < _layerSizes.Length - 1; layerIndex++)
        {
            for (var currentLayerNodeIndex = 0; currentLayerNodeIndex < _layerSizes[layerIndex]; currentLayerNodeIndex++)
            {
                for (var nextLayerNodeIndex = 0; nextLayerNodeIndex < _layerSizes[layerIndex + 1]; nextLayerNodeIndex++)
                {
                    Weights[weightAndBiasIndex] += delta;

                    var lossAfter = loss();

                    var deltaLoss = lossAfter - lossBefore;

                    var gradient = deltaLoss / delta;

                    weightGradients[weightAndBiasIndex] = gradient < 0 ? learnRate : learnRate * -1;

                    Weights[weightAndBiasIndex] -= delta;

                    Biases[weightAndBiasIndex] += delta;

                    lossAfter = loss();

                    deltaLoss = lossAfter - lossBefore;

                    gradient = deltaLoss / delta;

                    biasGradients[weightAndBiasIndex] = gradient < 0 ? learnRate : learnRate * -1;

                    Biases[weightAndBiasIndex] -= delta;

                    weightAndBiasIndex++;
                }
            }
        }

        // Update weights and biases based on gradients
        var gpuData = accelerator.Allocate1D(weightGradients.Concat(biasGradients).ToArray());

        var gpuOutput = accelerator.Allocate1D(Weights.Concat(Biases).ToArray());

        var loadedKernel = accelerator.LoadAutoGroupedStreamKernel((Index1D index, ArrayView<double>  data, ArrayView<double> output) =>
        {
            output[index] += data[index];
        });

        loadedKernel((int)gpuData.Length, gpuData.View, gpuOutput.View);

        accelerator.Synchronize();

        var hostOutput = gpuOutput.GetAsArray1D();

        weightAndBiasIndex = 0;
        for (var layerIndex = 0; layerIndex < _layerSizes.Length - 1; layerIndex++)
        {
            for (var currentLayerNodeIndex = 0; currentLayerNodeIndex < _layerSizes[layerIndex]; currentLayerNodeIndex++)
            {
                for (var nextLayerNodeIndex = 0; nextLayerNodeIndex < _layerSizes[layerIndex + 1]; nextLayerNodeIndex++)
                {
                    Weights[weightAndBiasIndex] = hostOutput[weightAndBiasIndex];
                    Biases[weightAndBiasIndex] = hostOutput[Weights.Length + weightAndBiasIndex];
                }
            }
        }
    }
}