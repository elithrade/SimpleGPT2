namespace SimpleGPT2;

public class Attention(int nHeads, int dModel)
{
    public Tensor Wq = new([dModel, dModel]);
    public Tensor Wk = new([dModel, dModel]);
    public Tensor Wv = new([dModel, dModel]);
    public Tensor Wo = new([dModel, dModel]);
    public Tensor Bq = new([dModel]);
    public Tensor Bk = new([dModel]);
    public Tensor Bv = new([dModel]);
    public Tensor Bo = new([dModel]);

    public Tensor Forward(Tensor x)
    {
        int T = x.Rows;
        int dHead = dModel / nHeads;

        // Step 1: project x into Q, K, V — each shape [T, dModel]
        Tensor Q = LinearWithBias(x, Wq, Bq);
        Tensor K = LinearWithBias(x, Wk, Bk);
        Tensor V = LinearWithBias(x, Wv, Bv);

        // Step 2: run attention independently for each head
        var headOutputs = new List<Tensor>();
        for (int h = 0; h < nHeads; h++)
        {
            int start = h * dHead;

            // Slice out this head's portion — each [T, dHead]
            Tensor q = Slice(Q, start, dHead);
            Tensor k = Slice(K, start, dHead);
            Tensor v = Slice(V, start, dHead);

            // scores[i,j] = how much token i attends to token j — shape [T, T]
            Tensor scores = MathOps.MatMul(q, Transpose(k));

            // Scale to prevent large dot products from saturating softmax
            float scale = 1.0f / MathF.Sqrt(dHead);
            for (int i = 0; i < scores.Data.Length; i++)
                scores.Data[i] *= scale;

            // Causal mask: token i cannot attend to token j if j > i (future)
            for (int i = 0; i < T; i++)
                for (int j = i + 1; j < T; j++)
                    scores[i, j] = -1e9f;

            // Softmax each row → attention weights (each row sums to 1)
            Tensor weights = new([T, T]);
            for (int i = 0; i < T; i++)
            {
                Tensor row = GetRow(scores, i);
                Tensor prob = MathOps.Softmax(row);
                for (int j = 0; j < T; j++)
                    weights[i, j] = prob[j];
            }

            // Weighted sum of V — each token's output is a blend of all values
            headOutputs.Add(MathOps.MatMul(weights, v));
        }

        // Step 3: concatenate all head outputs → [T, dModel], then project
        Tensor concat = Concat(headOutputs, T, dModel);
        return LinearWithBias(concat, Wo, Bo);
    }

    // x: [T, dModel], W: [dModel, dModel], b: [dModel] → [T, dModel]
    static Tensor LinearWithBias(Tensor x, Tensor w, Tensor b)
    {
        Tensor result = MathOps.MatMul(x, w);
        for (int i = 0; i < result.Rows; i++)
            for (int j = 0; j < result.Cols; j++)
                result[i, j] += b[j];
        return result;
    }

    // Extract columns [start .. start+size) from x → [T, size]
    static Tensor Slice(Tensor x, int start, int size)
    {
        int T = x.Rows;
        var result = new Tensor([T, size]);
        for (int i = 0; i < T; i++)
            for (int j = 0; j < size; j++)
                result[i, j] = x[i, start + j];
        return result;
    }

    // Transpose [T, D] → [D, T]
    static Tensor Transpose(Tensor x)
    {
        var result = new Tensor([x.Cols, x.Rows]);
        for (int i = 0; i < x.Rows; i++)
            for (int j = 0; j < x.Cols; j++)
                result[j, i] = x[i, j];
        return result;
    }

    // Extract a single row from a 2D tensor → 1D tensor
    static Tensor GetRow(Tensor x, int row)
    {
        int N = x.Cols;
        var result = new Tensor([N]);
        for (int j = 0; j < N; j++)
            result[j] = x[row, j];
        return result;
    }

    // Concatenate list of [T, dHead] tensors along columns → [T, dModel]
    static Tensor Concat(List<Tensor> heads, int T, int dModel)
    {
        var result = new Tensor([T, dModel]);
        int dHead = dModel / heads.Count;
        for (int h = 0; h < heads.Count; h++)
            for (int i = 0; i < T; i++)
                for (int j = 0; j < dHead; j++)
                    result[i, h * dHead + j] = heads[h][i, j];
        return result;
    }
}
