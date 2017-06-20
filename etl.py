import random
import re
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import unicodedata
from language import Language
from torch.autograd import Variable


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Scale(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])
cap = dset.CocoCaptions(root='data/train2014',
                        annFile='data/annotations/captions_train2014.json',
                        transform=transform)
n_examples = len(cap)


def get_example(lang, encoder, i):
    img, target = cap[i % n_examples]

    img = Variable(img).view(1, 512, 512, 3).cuda()
    img = encoder.features[28](img)
    target = normalize_string(random.choice(target))
    target = variable_from_sentence(lang, target)

    return img, target


# Returns a list of indexes, one for each word in the sentence
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def prepare_data():
    lang = Language('eng')

    for i in range(100):
        image, captions = cap[i]
        normed_captions = [normalize_string(caption) for caption in captions]

        for c in normed_captions:
            lang.index_words(c)

    return lang


# Turns a unicode string to plain ASCII (http://stackoverflow.com/a/518232/2809427)
def unicode_to_ascii(s):
    chars = [c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn']
    char_list = ''.join(chars)
    return char_list


def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(1)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    var = var.cuda()
    return var
