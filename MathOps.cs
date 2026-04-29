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
}
