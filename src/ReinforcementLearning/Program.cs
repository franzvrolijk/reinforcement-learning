using ILGPU;
using System.Diagnostics;

namespace ReinforcementLearning;

public class Program
{
    private static void Main()
    {
        const int trainingIterations = 1000;
        const double initialLearningRate = 0.001d;
        const double learningRateDecay = 0.001d;
        const double delta = 0.00000001d;

        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(false).CreateAccelerator(context);

        var network = new Network([4, 4, 1], Activation.Sigmoid);
        var iteration = 1;

        var testData = Enumerable.Range(0, 10).Select(_ => new double[] { Random.Shared.NextDouble(), Random.Shared.NextDouble(), Random.Shared.NextDouble(), Random.Shared.NextDouble() }).ToArray();

        while (true)
        {
            var learningRate = initialLearningRate / (1 + learningRateDecay * iteration);

            var s = Stopwatch.StartNew();

            var inputs = new double[] { Random.Shared.NextDouble(), Random.Shared.NextDouble(), Random.Shared.NextDouble(), Random.Shared.NextDouble() };

            var target = inputs.Sum() / 4d;

            for (var i = 0; i < trainingIterations; i++)
            {
                network.GradientDescentGpu(() =>
                {
                    var outputs = network.Propagate(inputs)[0];
                    var error = target - outputs;
                    var loss = error * error;
                    return loss;
                }, learningRate, delta, accelerator);
            }

            s.Stop();

            if (Console.KeyAvailable)
            {
                List<double> scores = [];

                for (var i = 0; i < 10; i++)
                {
                    var testCase = testData[i];
                    var outputs = network.Propagate(testCase)[0];
                    var error = target - outputs;
                    var loss = error * error;

                    scores.Add(1 - loss);
                }

                var min = scores.Min();
                var max = scores.Max();
                var (avg, spread) = (scores.Average(), max - min);

                Console.ReadKey(true);
                //Console.Clear();
                //network.Print();
                Console.WriteLine($"(Iteration {iteration})\tAvg: {avg:0.00}\tMin: {min:0.00}\tMax {max:0.00}\tSpread: {spread:0.00}\tLR: {learningRate:0.000000}\tTime: {s.ElapsedMilliseconds}ms");
            }

            iteration++;
        }
    }
}