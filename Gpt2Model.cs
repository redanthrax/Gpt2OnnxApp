using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

namespace Gpt2OnnxApp {
    public class Gpt2Model {
        private readonly InferenceSession _session;
        private readonly Gpt2Tokenizer _tokenizer;

        public Gpt2Model(string modelPath, string vocabPath) {
            _session = new InferenceSession(modelPath);
            _tokenizer = new Gpt2Tokenizer(vocabPath);
        }

        public string RunInference(string input) {
            var (tokens, attentionMask) = _tokenizer.Tokenize(input);

            var inputs = new List<NamedOnnxValue>
            {
            NamedOnnxValue.CreateFromTensor("input1", tokens),
        };

            using var results = _session.Run(inputs);
            var outputTokenTensor = results.First().AsEnumerable<long>().ToArray();
            var outputTensor = new DenseTensor<long>(outputTokenTensor, new[] { 1, outputTokenTensor.Length, 1 });
            return _tokenizer.Detokenize(outputTensor);
        }
    }
}
