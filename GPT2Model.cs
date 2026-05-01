namespace SimpleGPT2;

public class GPT2Model
{
    const int NLayers = 12;
    const int NHeads = 12;
    const int DModel = 768;
    const int VocabSize = 50257;
    const int MaxSeqLen = 1024;

    // Embeddings
    public Tensor Wte = new([VocabSize, DModel]); // token embeddings
    public Tensor Wpe = new([MaxSeqLen, DModel]); // positional embeddings

    // 12 transformer blocks
    public TransformerBlock[] Blocks =
    [
        .. Enumerable.Range(0, NLayers).Select(_ => new TransformerBlock(NHeads, DModel)),
    ];

    // Final layer norm
    public Tensor LnFW = new([DModel]);
    public Tensor LnFB = new([DModel]);

    public Tensor Forward(int[] tokenIds)
    {
        int T = tokenIds.Length;

        // Step 1: build input [T, DModel] by adding token + positional embeddings
        var x = new Tensor([T, DModel]);
        for (int i = 0; i < T; i++)
        for (int j = 0; j < DModel; j++)
            // TODO: x[i,j] = token embedding for tokenIds[i] + positional embedding for position i
            x[i, j] = Wte[tokenIds[i], j] + Wpe[i, j];

        // Step 2: pass through each transformer block
        // TODO: foreach block, x = block.Forward(x)
        foreach (var block in Blocks)
            x = block.Forward(x);

        // Step 3: final LayerNorm on each row
        // TODO: same row-by-row LayerNorm pattern as TransformerBlock
        for (int i = 0; i < T; i++)
        {
            // extract row i into a 1D tensor
            var row = new Tensor([DModel]);
            for (int j = 0; j < DModel; j++)
                row[j] = x[i, j];

            // LayerNorm that row
            var ln = MathOps.LayerNorm(row, LnFW, LnFB);

            // put it back into x
            for (int j = 0; j < DModel; j++)
                x[i, j] = ln[j];
        }

        // Step 4: project to logits — [T, DModel] × [DModel, VocabSize] → [T, VocabSize]
        // GPT-2 reuses Wte transposed as the output projection (weight tying)
        Tensor WteT = Transpose(Wte); // [DModel, VocabSize]
        return MatMul(x, WteT); // [T, VocabSize]
    }

    static Tensor Transpose(Tensor x)
    {
        // same as in Attention
        var result = new Tensor([x.Cols, x.Rows]);
        for (int i = 0; i < x.Rows; i++)
        for (int j = 0; j < x.Cols; j++)
            result[j, i] = x[i, j];

        return result;
    }

    static Tensor MatMul(Tensor a, Tensor b) => MathOps.MatMul(a, b);
}
