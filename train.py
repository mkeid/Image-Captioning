import argparse
import etl
import helpers
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from decoder import DecoderRNN
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--dropout_p', default=.05)
parser.add_argument('--epochs', default=200000)
parser.add_argument('--grad_clip', default=10.)
parser.add_argument('--learning_rate', default=.000015625)
parser.add_argument('--plot_every', default=200)
parser.add_argument('--print_every', default=100)
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--save_every', default=5000)
parser.add_argument('--teacher_forcing_ratio', default=.5)
args = parser.parse_args()

n_layers = 2
hidden_size = 448


def train(input_var, target_var, decoder, decoder_opt, criterion, caption=''):
    # Initialize optimizer and loss
    decoder_opt.zero_grad()
    loss = 0.

    # Get target sequence length
    target_length = target_var.size()[0]

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([0]))
    decoder_input = decoder_input.cuda()
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_context = decoder_context.cuda()
    decoder_h = decoder.init_hidden()
    decoder_c = decoder.init_hidden()

    decoded_words = []

    # Scheduled sampling
    use_teacher_forcing = random.random() < args.teacher_forcing_ratio
    if use_teacher_forcing:
        # Feed target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_h, decoder_c, decoder_attention = decoder(decoder_input,
                                                                                    decoder_context,
                                                                                    decoder_h,
                                                                                    decoder_c,
                                                                                    input_var)
            loss += criterion(decoder_output[0], target_var[di])
            decoder_input = target_var[di]

            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == 1:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang.index2word[ni])

    else:
        # Use previous prediction as next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_h, decoder_c, decoder_attention = decoder(decoder_input,
                                                                                         decoder_context,
                                                                                         decoder_h,
                                                                                         decoder_c,
                                                                                         input_var)

            loss += criterion(decoder_output[0], target_var[di])
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda()

            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == 1:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang.index2word[ni])

            if ni == 1:
                break

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(decoder.parameters(), args.grad_clip)
    decoder_opt.step()

    output_sentence = ' '.join(decoded_words)
    print("---\nPredicted: %s\nActual: %s" % (output_sentence, caption))
    return loss.data[0] / target_length

# Initialize models
lang = etl.prepare_data()
encoder = models.vgg16(pretrained=True)
decoder = DecoderRNN('general', hidden_size, lang.n_words, n_layers, dropout_p=args.dropout_p)

# Make sure we do not train our encoder net
for param in encoder.parameters():
    param.requires_grad = False

if args.retrain:
    decoder.load_state_dict(torch.load('data/decoder_params'))
    decoder.attention.load_state_dict(torch.load('data/attention_params'))

# Move models to GPU
encoder.cuda()
decoder.cuda()

# Initialize optimizers and criterion
decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.NLLLoss()

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every

# Begin training
for epoch in range(1, args.epochs + 1):

    # Get training data for this cycle
    input_variable, target_variable, caption = etl.get_example(lang, encoder, epoch - 1)

    # Run the train step
    loss = train(input_variable, target_variable, decoder, decoder_optimizer, criterion, caption)

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss

    if epoch == 0:
        continue

    # Occasionally print training status
    if epoch % args.print_every == 0:
        print_loss_avg = print_loss_total / args.print_every
        print_loss_total = 0
        time_since = helpers.time_since(start, epoch / args.epochs)
        print('%s (%d %d%%) %.4f' % (time_since, epoch, epoch / args.epochs * 100, print_loss_avg))

    # Occasionally plot loss
    if epoch % args.plot_every == 0:
        plot_loss_avg = plot_loss_total / args.plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

    # Occasionally save models
    if epoch % args.save_every == 0:
        torch.save(decoder.state_dict(), 'data/decoder_params')
        torch.save(decoder.attention.state_dict(), 'data/attention_params')
        helpers.save_object(lang, 'language')

# Save our models
print("Saving models...")
torch.save(decoder.state_dict(), 'data/decoder_params')
torch.save(decoder.attention.state_dict(), 'data/attention_params')
print("Models have been saved to the data directory.")

# Plot loss
helpers.show_plot(plot_losses)

