using Gpt2OnnxApp;
using Tokenizers.DotNet;

var modelPath = "gpt2-10.onnx";
var hubName = "openai-community/gpt2";
var filePath = "tokenizer.json";
var vocabPath = await HuggingFace.GetFileFromHub(hubName, filePath, "deps");

var model = new Gpt2Model(modelPath, vocabPath);

Console.WriteLine("Enter input text:");
var input = Console.ReadLine();

var output = model.RunInference(input);
Console.WriteLine("Model output:");
Console.WriteLine(output);