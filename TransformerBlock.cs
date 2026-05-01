namespace SimpleGPT2;

public class TransformerBlock(int nHeads, int dModel)
{
    public Attention Attn = new(nHeads, dModel);

    public Tensor Ln1W = new([dModel]);
    public Tensor Ln1B = new([dModel]);
    public Tensor Ln2W = new([dModel]);
    public Tensor Ln2B = new([dModel]);

    public Tensor FfnW1 = new([dModel, 4 * dModel]);
    public Tensor FfnB1 = new([4 * dModel]);
    public Tensor FfnW2 = new([4 * dModel, dModel]);
    public Tensor FfnB2 = new([dModel]);

    public Tensor Forward(Tensor x)
    {
        int T = x.Rows;

        // TODO 1: x = x + Attention(LayerNorm(x, Ln1W, Ln1B))
        // Apply LayerNorm row by row, run through attention, add residual
        // LayerNorm each row → build normalized [T, dModel] tensor
        var normed = new Tensor([T, dModel]);

        for (int i = 0; i < T; i++)
        {
            var row = new Tensor([dModel]);
            for (int j = 0; j < dModel; j++)
                row[j] = x[i, j];

            // LayerNorm that row
            var ln = MathOps.LayerNorm(row, Ln1W, Ln1B);
            // Put it back into normed
            for (int j = 0; j < dModel; j++)
                normed[i, j] = ln[j];
        }

        // Run attention on normed tensor, then add residual
        x = AddResidual(x, Attn.Forward(normed));

        var normed2 = new Tensor([T, dModel]);
        for (int i = 0; i < T; i++)
        {
            // extract row i
            var row = new Tensor([dModel]);
            for (int j = 0; j < dModel; j++)
                row[j] = x[i, j];

            // LayerNorm
            var ln = MathOps.LayerNorm(row, Ln2W, Ln2B);

            // FFN: Linear → GELU → Linear
            Tensor h = MathOps.MatMul(new Tensor(ln.Data, [1, dModel]), FfnW1);
            for (int j = 0; j < h.Cols; j++)
                h[0, j] += FfnB1[j]; // add bias

            Tensor hGelu = MathOps.Gelu(h);

            Tensor out2 = MathOps.MatMul(hGelu, FfnW2);
            for (int j = 0; j < out2.Cols; j++)
                out2[0, j] += FfnB2[j]; // add bias

            for (int j = 0; j < dModel; j++)
                normed2[i, j] = out2[0, j];
        }

        x = AddResidual(x, normed2);

        return x;
    }

    static Tensor AddResidual(Tensor a, Tensor b)
    {
        var result = new Tensor(a.Shape);
        for (int i = 0; i < a.Data.Length; i++)
            result.Data[i] = a.Data[i] + b.Data[i];
        return result;
    }
}
