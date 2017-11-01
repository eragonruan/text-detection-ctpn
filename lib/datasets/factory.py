__sets = {}
from .pascal_voc import pascal_voc

def _selective_search_IJCV_top_k(split, year, top_k):
    imdb = pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb
# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012', '0712']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                pascal_voc(split, year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        print((list_imdbs()))
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
