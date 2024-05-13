namespace ReinforcementLearning;

public class Activation
{
    public static double Sigmoid(double value)
    {
        var x = Math.Exp(-value);

        if (double.IsNaN(x))
        {
            throw new("NaN encountered");
        }

        return 1 / (1 + x);
    }

    public static double[] Softmax(double[] values)
    {
        var max = values.Max();
        var scale = 0.0;

        for (var i = 0; i < values.Length; i++)
        {
            values[i] = Math.Exp(values[i] - max);
            scale += values[i];
        }

        for (var i = 0; i < values.Length; i++)
        {
            values[i] /= scale;
        }

        return values;
    }
}