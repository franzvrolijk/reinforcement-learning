using System.Text;
using ILGPU;
using ILGPU.Runtime;

namespace ReinforcementLearning;

public class Network
{
    private readonly double[] _nodes;

    public double[] Weights;

    public double[] Biases;

    private readonly int[] _layerSizes;

    private readonly Func<double, double> _activationFunction;
    private readonly Func<double, double> _outputActivation;
    private static double RandomWeightValue => 0.01 * Random.Shared.NextDouble() - 0.005;

    public Network(int[] layerSizes, Func<double, double> activationFunction, Func<double, double> outputActivation, double[]? weights = null, double[]? biases = null)
    {
        _layerSizes = layerSizes;
        _activationFunction = activationFunction;
        _outputActivation = outputActivation;

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
        return new(_layerSizes, _activationFunction, _outputActivation, Weights.Select(w => w).ToArray(), Biases.Select(b => b).ToArray());
    }

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
            var nextLayerStartIndex = nodeIndex + _layerSizes[layerIndex];
            for (var currentNodeIndex = 0; currentNodeIndex < _layerSizes[layerIndex]; currentNodeIndex++)
            {
                for (var nextNodeIndex = 0; nextNodeIndex < _layerSizes[layerIndex + 1]; nextNodeIndex++)
                {
                    // Calculate the index of the next node in the _nodes array
                    var nextNodeArrayIndex = nextLayerStartIndex + nextNodeIndex;

                    // Accumulate the weighted sum for each node in the next layer
                    _nodes[nextNodeArrayIndex] += _nodes[nodeIndex + currentNodeIndex] * Weights[weightAndBiasIndex] + Biases[weightAndBiasIndex];

                    weightAndBiasIndex++;
                }
            }

            var activation = layerIndex == _layerSizes.Length - 2
                ? _outputActivation
                : _activationFunction;

            // Apply activation function to each node in the next layer
            for (var nextNodeIndex = 0; nextNodeIndex < _layerSizes[layerIndex + 1]; nextNodeIndex++)
            {
                _nodes[nextLayerStartIndex + nextNodeIndex] = activation(_nodes[nextLayerStartIndex + nextNodeIndex]);
            }

            nodeIndex += _layerSizes[layerIndex];
        }

        var nodesBeforeOutput = _nodes.Length - _layerSizes[^1];

        // Calculate outputs
        return _nodes[nodesBeforeOutput..];
    }

    public void GradientDescent(Func<double> loss, double learnRate, double delta)
    {
        var weightUpdates = new double[Weights.Length];
        var biasUpdates = new double[Biases.Length];

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

                    weightUpdates[weightAndBiasIndex] = derivative < 0
                        ? learnRate
                        : learnRate * -1; ;

                    Weights[weightAndBiasIndex] -= delta;

                    lossBefore = lossAfter;

                    Biases[weightAndBiasIndex] += delta;

                    lossAfter = loss();

                    deltaLoss = lossAfter - lossBefore;

                    derivative = deltaLoss / delta;

                    biasUpdates[weightAndBiasIndex] = derivative < 0
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
                    Weights[weightAndBiasIndex] += weightUpdates[weightAndBiasIndex];
                    Biases[weightAndBiasIndex] += biasUpdates[weightAndBiasIndex];

                    weightAndBiasIndex++;
                }
            }
        }
    }

    public void GradientDescentGpu(Func<double> loss, double learnRate, double delta, Accelerator accelerator)
    {
        var weightUpdates = new double[Weights.Length];
        var biasUpdates = new double[Biases.Length];

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

                    weightUpdates[weightAndBiasIndex] = gradient < 0 ? learnRate : learnRate * -1;

                    Weights[weightAndBiasIndex] -= delta;

                    Biases[weightAndBiasIndex] += delta;

                    lossAfter = loss();

                    deltaLoss = lossAfter - lossBefore;

                    gradient = deltaLoss / delta;

                    biasUpdates[weightAndBiasIndex] = gradient < 0 ? learnRate : learnRate * -1;

                    Biases[weightAndBiasIndex] -= delta;

                    weightAndBiasIndex++;
                }
            }
        }

        // Update weights and biases based on gradients
        var weightAndBiasUpdates = weightUpdates.Concat(biasUpdates).ToArray();
        var weightsAndBiases = Weights.Concat(Biases).ToArray();

        var gpuData = accelerator.Allocate1D(weightAndBiasUpdates);

        var gpuOutput = accelerator.Allocate1D(weightsAndBiases);

        var loadedKernel = accelerator.LoadAutoGroupedStreamKernel((Index1D index, ArrayView<double> weightAndBiasUpdatesGpu, ArrayView<double> weightsAndBiasesGpu) =>
        {
            // This runs on GPU
            weightsAndBiasesGpu[index] += weightAndBiasUpdatesGpu[index];
        });

        loadedKernel((int)gpuData.Length, gpuData.View, gpuOutput.View);

        accelerator.Synchronize();

        var hostOutput = gpuOutput.GetAsArray1D();

        Buffer.BlockCopy(hostOutput, 0, Weights, 0, Weights.Length * sizeof(double));
        Buffer.BlockCopy(hostOutput, Weights.Length * sizeof(double), Biases, 0, Biases.Length * sizeof(double));
    }

    public void Print()
    {
        var weightAndBiasIndex = 0;
        for (var layerIndex = 0; layerIndex < _layerSizes.Length - 1; layerIndex++)
        {
            var stringBuilder = new StringBuilder();
            stringBuilder.Append($"L{layerIndex}-L{layerIndex + 1}\n");
            for (var currentLayerNodeIndex = 0; currentLayerNodeIndex < _layerSizes[layerIndex]; currentLayerNodeIndex++)
            {
                for (var nextLayerNodeIndex = 0; nextLayerNodeIndex < _layerSizes[layerIndex + 1]; nextLayerNodeIndex++)
                {
                    stringBuilder.AppendLine();
                    stringBuilder.Append($"""
                                         Node {currentLayerNodeIndex} to {nextLayerNodeIndex}
                                            Weight: {Weights[weightAndBiasIndex]}
                                            Bias: {Biases[weightAndBiasIndex]}
                                         """);
                    weightAndBiasIndex++;
                }
            }

            Console.WriteLine(stringBuilder.ToString());
        }
    }
}