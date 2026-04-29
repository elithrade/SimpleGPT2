namespace SimpleGPT2;

public static class MathOps
{
    public static Tensor MatMul(Tensor a, Tensor b)
    {
        // a: [M, K], b: [K, N] → result: [M, N]
        int M = a.Rows,
            K = a.Cols,
            N = b.Cols;
        var result = new Tensor([M, N]);

        for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            float sum = 0;
            // TODO: dot product of row i of a with column j of b
            for (int k = 0; k < K; k++)
            {
                sum += a[i, k] * b[k, j];
            }
            result[i, j] = sum;
        }

        return result;
    }

    public static Tensor Softmax(Tensor x)
    {
        // x is 1D: shape [N]
        int N = x.Data.Length;
        var result = new Tensor([N]);

        float max = x.Data.Max(); // stability trick

        float sum = 0;
        for (int i = 0; i < N; i++)
        {
            // TODO 1: result[i] = e^(x[i] - max)
            result[i] = MathF.Exp(x[i] - max);
            sum += result[i];
        }

        for (int i = 0; i < N; i++)
        {
            // TODO 2: divide result[i] by sum to normalize
            result[i] /= sum;
        }

        return result;
    }

    public static Tensor LayerNorm(Tensor x, Tensor w, Tensor b, float eps = 1e-5f)
    {
        int N = x.Data.Length;
        var result = new Tensor([N]);

        // TODO 1: compute mean of x
        float mean = x.Data.Average();

        // TODO 2: compute variance of x
        float variance = x.Data.Average(v => (v - mean) * (v - mean));

        // TODO 3: normalize, scale and shift
        for (int i = 0; i < N; i++)
            result[i] = (x[i] - mean) / MathF.Sqrt(variance + eps) * w[i] + b[i];

        return result;
    }

    public static Tensor Gelu(Tensor x)
    {
        int N = x.Data.Length;
        var result = new Tensor([N]);

        float c = MathF.Sqrt(2.0f / MathF.PI); // constant √(2/π)

        for (int i = 0; i < N; i++)
        {
            float v = x[i];
            // GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
            // TODO: apply the formula above
            result[i] = 0.5f * v * (1 + MathF.Tanh(c * (v + 0.044715f * MathF.Pow(v, 3))));
        }

        return result;
    }
}
