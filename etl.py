import helpers
import os
import PIL.Image as Image
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
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])
cap = dset.CocoCaptions(root='data/train2014',
                        annFile='data/annotations/captions_train2014.json',
                        transform=transform)
n_examples = len(cap)


def get_example(lang, encoder, i):

    # Retrieve next sample
    img, caption = cap[i % n_examples]

    # Transform image
    img = Variable(img).view(1, 3, 224, 224).cuda()
    for i in range(30):
        img = encoder.features[i](img)
    img = img.view(256, -1)

    # Transform annotation
    caption = normalize_string(random.choice(caption))
    target = variable_from_sentence(lang, caption)

    return img, target, caption


def get_image_from_path(p, enc):
    p = os.path.expanduser(p)
    img = Image.open(p)
    img = transform(img)
    img = Variable(img).cuda()
    img = img.unsqueeze(0)
    for i in range(30):
        img = enc.features[i](img)
    img = img.view(256, -1)
    return img


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
    if os.path.exists('data/language'):
        lang = helpers.load_object('language')

    else:
        lang = Language('eng')

        for i in range(n_examples):
            image, captions = cap[i]
            normed_captions = [normalize_string(caption) for caption in captions]

            for c in normed_captions:
                lang.index_words(c)

        helpers.save_object(lang, 'language')

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
