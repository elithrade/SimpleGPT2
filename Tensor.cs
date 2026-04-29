namespace SimpleGPT2;

public class Tensor
{
    public float[] Data { get; }
    public int[] Shape { get; }

    public Tensor(int[] shape)
    {
        Shape = shape;
        Data = new float[shape.Aggregate(1, (a, b) => a * b)];
    }

    public Tensor(float[] data, int[] shape)
    {
        // TODO 1: validate data.Length == product of all dims in shape
        if (data.Length != shape.Aggregate(1, (a, b) => a * b))
            throw new ArgumentException("Data length does not match shape dimensions.");
        // TODO 2: assign Data and Shape
        Data = data;
        Shape = shape;
    }

    public float this[int i, int j]
    {
        // TODO 3: row-major indexing — element [i,j] is at Data[i * Shape[1] + j]
        get => Data[i * Shape[1] + j];
        set => Data[i * Shape[1] + j] = value;
    }

    public float this[int i]
    {
        get => Data[i];
        set => Data[i] = value;
    }

    public int Rows => Shape[0];
    public int Cols => Shape[^1];

    public override string ToString() => $"Tensor({string.Join("x", Shape)})";
}
