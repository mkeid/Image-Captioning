import argparse
import etl
import helpers
import torch
import torchvision.models as models
from decoder import DecoderRNN
from language import Language
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()
helpers.validate_path(args.path)

# Parse argument for input sentence
parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()
helpers.validate_path(args.path)

n_layers = 2

# Initialize models
lang = helpers.load_object('language')
encoder = models.vgg16(pretrained=True)
decoder = DecoderRNN('general', 512, lang.n_words, n_layers, dropout_p=0.)

# Load model parameters
decoder.load_state_dict(torch.load('data/decoder_params'))
decoder.attention.load_state_dict(torch.load('data/attention_params'))

# Move models to GPU
encoder.cuda()
decoder.cuda()


def evaluate(path, encoder, max_length=256):
    input_variable = etl.get_image_from_path(path, encoder)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[0]]))
    decoder_input = decoder_input.cuda()
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_context = decoder_context.cuda()
    decoder_hidden = decoder.init_hidden()

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                                     decoder_context,
                                                                                     decoder_hidden,
                                                                                     input_variable)

        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == Language.eos_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[ni])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda()

        return decoded_words, decoder_attentions[:di + 1, :len(input_variable)]


output_words, decoder_attn = evaluate(args.path, encoder)
output_sentence = ' '.join(output_words)
print(output_sentence)
