
using Microsoft.ML.OnnxRuntime.Tensors;
using Tokenizers.DotNet;

namespace Gpt2OnnxApp {
    public class Gpt2Tokenizer {
        private readonly Tokenizer _tokenizer;

        public Gpt2Tokenizer(string vocabPath) {
            _tokenizer = new Tokenizer(vocabPath: vocabPath);
        }

        public (Tensor<long>, Tensor<long>) Tokenize(string input) {
            var tokens = _tokenizer.Encode(input);
            var tokenIds = tokens.Select(id => (long)id).ToArray();
            var attentionMask = tokenIds.Select(_ => 1L).ToArray();

            return (new DenseTensor<long>(tokenIds, new[] { 1, tokenIds.Length, 1 }),
                    new DenseTensor<long>(attentionMask, new[] { 1, attentionMask.Length, 1 }));
        }

        public string Detokenize(Tensor<long> tokenTensor) {
            var tokenIds = tokenTensor.ToArray().Select(t => (uint)t).ToArray();
            return _tokenizer.Decode(tokenIds);
        }
    }
}
