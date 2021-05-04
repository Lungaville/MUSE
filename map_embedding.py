import numpy as np
import torch
import io
import os
from torch.autograd import Variable

from src.dictionary import Dictionary

class Dynamic(object):
    pass

def reload_best(params):
        """
        Reload the best mapping.
        """
        path = params.mapper_location
        print('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        W = mapping.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))
        print(W)
def load_embeddings(params, source, full_vocab=False):
    """
    Reload pretrained embeddings.
    - `full_vocab == False` means that we load the `params.max_vocab` most frequent words.
      It is used at the beginning of the experiment.
      In that setting, if two words with a different casing occur, we lowercase both, and
      only consider the most frequent one. For instance, if "London" and "london" are in
      the embeddings file, we only consider the most frequent one, (in that case, probably
      London). This is done to deal with the lowercased dictionaries.
    - `full_vocab == True` means that we load the entire embedding text file,
      before we export the embeddings at the end of the experiment.
    """
    assert type(source) is bool and type(full_vocab) is bool
    emb_path = params.src_emb if source else params.tgt_emb
    if emb_path.endswith('.pth'):
        return load_pth_embeddings(params, source, full_vocab)
    if emb_path.endswith('.bin'):
        return load_bin_embeddings(params, source, full_vocab)
    else:
        return read_txt_embeddings(params, source, full_vocab)


def export(params):
        params = params

        # load all embeddings
        print("Reloading all embeddings for mapping ...")
        params.src_dico, src_emb = load_embeddings(params, source=True, full_vocab=False)
        params.tgt_dico, tgt_emb = load_embeddings(params, source=False, full_vocab=False)

        # apply same normalization as during training
        normalize_embeddings(src_emb, normalize_type)
        normalize_embeddings(tgt_emb, normalize_type)

        # map source embeddings to the target space
        bs = 4096
        print("Map source embeddings to the target space ...")
        for i, k in enumerate(range(0, len(src_emb), bs)):
            x = Variable(src_emb[k:k + bs], volatile=True)
            src_emb[k:k + bs] = mapping(x.cuda() if params.cuda else x).data.cpu()

        # write embeddings to the disk
        export_embeddings(src_emb, tgt_emb, params)

def read_txt_embeddings(params, source, full_vocab):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    # load pretrained embeddings
    lang = params.src_lang if source else params.tgt_lang
    emb_path = params.src_emb if source else params.tgt_emb
    _emb_dim_file = params.emb_dim
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                assert _emb_dim_file == int(split[1])
            else:
                word, vect = line.rstrip().split(' ', 1)
                if not full_vocab:
                    word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                if word in word2id:
                    if full_vocab:
                        print("Word '%s' found twice in %s embedding file"
                                       % (word, 'source' if source else 'target'))
                else:
                    if not vect.shape == (_emb_dim_file,):
                        print("Invalid dimension (%i) for %s word '%s' in line %i."
                                       % (vect.shape[0], 'source' if source else 'target', word, i))
                        continue
                    assert vect.shape == (_emb_dim_file,), i
                    word2id[word] = len(word2id)
                    vectors.append(vect[None])
            if params.max_vocab > 0 and len(word2id) >= params.max_vocab and not full_vocab:
                break


    assert len(word2id) == len(vectors)
    print("Loaded %i pre-trained word embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)
    embeddings = np.concatenate(vectors, 0)
    embeddings = torch.from_numpy(embeddings).float()
    embeddings = embeddings.cuda() if (params.cuda and not full_vocab) else embeddings

    assert embeddings.size() == (len(dico), params.emb_dim)
    return dico, embeddings

def normalize_embeddings(emb, types, mean=None):
    """
    Normalize embeddings by their norms / recenter them.
    """
    for t in types.split(','):
        if t == '':
            continue
        if t == 'center':
            if mean is None:
                mean = emb.mean(0, keepdim=True)
            emb.sub_(mean.expand_as(emb))
        elif t == 'renorm':
            emb.div_(emb.norm(2, 1, keepdim=True).expand_as(emb))
        else:
            raise Exception('Unknown normalization type: "%s"' % t)
    return mean.cpu() if mean is not None else None

def export_embeddings(src_emb, tgt_emb, params):
    """
    Export embeddings to a text or a PyTorch file.
    """
    assert params.export in ["txt", "pth"]

    # text file
    if params.export == "txt":
        src_path = os.path.join(params.exp_path, 'vectors-%s.txt' % params.src_lang)
        tgt_path = os.path.join(params.exp_path, 'vectors-%s.txt' % params.tgt_lang)
        # source embeddings
        print('Writing source embeddings to %s ...' % src_path)
        with io.open(src_path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % src_emb.size())
            for i in range(len(params.src_dico)):
                f.write(u"%s %s\n" % (params.src_dico[i], " ".join('%.5f' % x for x in src_emb[i])))
        # target embeddings
        print('Writing target embeddings to %s ...' % tgt_path)
        with io.open(tgt_path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % tgt_emb.size())
            for i in range(len(params.tgt_dico)):
                f.write(u"%s %s\n" % (params.tgt_dico[i], " ".join('%.5f' % x for x in tgt_emb[i])))

    # PyTorch file
    if params.export == "pth":
        src_path = os.path.join(params.exp_path, 'vectors-%s.pth' % params.src_lang)
        tgt_path = os.path.join(params.exp_path, 'vectors-%s.pth' % params.tgt_lang)
        print('Writing source embeddings to %s ...' % src_path)
        torch.save({'dico': params.src_dico, 'vectors': src_emb}, src_path)
        print('Writing target embeddings to %s ...' % tgt_path)
        torch.save({'dico': params.tgt_dico, 'vectors': tgt_emb}, tgt_path)

parser = argparse.ArgumentParser(description='Map Embedding')
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
# data
parser.add_argument("--src_lang", type=str, default="", help="Source language")
parser.add_argument("--tgt_lang", type=str, default="", help="Target language")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--mapper_location", type=str, default="", help="Reload mapper")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")

# parse parameters
params = parser.parse_args()

normalize_type = ""
mapping = torch.nn.Linear(params.emb_dim, params.emb_dim, bias=False).cuda()
reload_best(params)
export(params)